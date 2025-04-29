import json
import os
import shutil

import torch
from loguru import logger
from mser import __version__


def load_pretrained(model, pretrained_model):
    """Load a pretrained model

    :param model: Model instance
    :param pretrained_model: Path to pretrained model
    """
    # Load pretrained model
    if pretrained_model is None:
        return model
    if os.path.isdir(pretrained_model):
        pretrained_model = os.path.join(pretrained_model, 'model.pth')
    assert os.path.exists(pretrained_model), f"Model {pretrained_model} does not exist!"
    
    model_state_dict = torch.load(pretrained_model, weights_only=False)

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_dict = model.module.state_dict()
    else:
        model_dict = model.state_dict()

    # Filter out incompatible parameters
    for name, weight in model_dict.items():
        if name in model_state_dict.keys():
            if list(weight.shape) != list(model_state_dict[name].shape):
                logger.warning(f'{name} not used, shape {list(model_state_dict[name].shape)} '
                               f'unmatched with {list(weight.shape)} in model.')
                model_state_dict.pop(name, None)

    # Load weights
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        missing_keys, unexpected_keys = model.module.load_state_dict(model_state_dict, strict=False)
    else:
        missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)

    if len(unexpected_keys) > 0:
        logger.warning('Unexpected key(s) in state_dict: {}.'
                       .format(', '.join('"{}"'.format(k) for k in unexpected_keys)))
    if len(missing_keys) > 0:
        logger.warning('Missing key(s) in state_dict: {}.'
                       .format(', '.join('"{}"'.format(k) for k in missing_keys)))

    logger.info('Successfully loaded pretrained model: {}'.format(pretrained_model))
    return model


def load_checkpoint(configs, model, optimizer, amp_scaler, scheduler,
                    step_epoch, save_model_path, resume_model):
    """Load model checkpoint

    :param configs: Configuration settings
    :param model: Model instance
    :param optimizer: Optimizer instance
    :param amp_scaler: Automatic Mixed Precision (AMP) scaler
    :param scheduler: Learning rate scheduler
    :param step_epoch: Number of steps per epoch
    :param save_model_path: Path where models are saved
    :param resume_model: Path to resume model from
    """
    last_epoch1 = 0
    accuracy1 = 0.

    def load_model(model_path):
        assert os.path.exists(os.path.join(model_path, 'model.pth')), "Model parameters file does not exist!"
        assert os.path.exists(os.path.join(model_path, 'optimizer.pth')), "Optimizer parameters file does not exist!"
        state_dict = torch.load(os.path.join(model_path, 'model.pth'), weights_only=False)

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)

        optimizer.load_state_dict(torch.load(os.path.join(model_path, 'optimizer.pth'), weights_only=False))

        # Load AMP scaler if available
        if amp_scaler is not None and os.path.exists(os.path.join(model_path, 'scaler.pth')):
            amp_scaler.load_state_dict(torch.load(os.path.join(model_path, 'scaler.pth'), weights_only=False))

        with open(os.path.join(model_path, 'model.state'), 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            last_epoch = json_data['last_epoch']
            accuracy = json_data['accuracy']

        logger.info('Successfully restored model and optimizer parameters from: {}'.format(model_path))
        
        optimizer.step()
        [scheduler.step() for _ in range(last_epoch * step_epoch)]

        return last_epoch, accuracy

    # Determine the last model directory
    save_feature_method = configs.preprocess_conf.feature_method
    if configs.preprocess_conf.get('use_hf_model', False):
        save_feature_method = save_feature_method.rstrip('/')
        save_feature_method = os.path.basename(save_feature_method)

    last_model_dir = os.path.join(save_model_path,
                                  f'{configs.model_conf.model}_{save_feature_method}',
                                  'last_model')

    if resume_model is not None or (os.path.exists(os.path.join(last_model_dir, 'model.pth'))
                                    and os.path.exists(os.path.join(last_model_dir, 'optimizer.pth'))):
        if resume_model is not None:
            last_epoch1, accuracy1 = load_model(resume_model)
        else:
            try:
                last_epoch1, accuracy1 = load_model(last_model_dir)
            except Exception as e:
                logger.warning(f'Failed to automatically resume the latest model, error: {e}')
    return model, optimizer, amp_scaler, scheduler, last_epoch1, accuracy1


def save_checkpoint(configs, model, optimizer, amp_scaler, save_model_path, epoch_id,
                    accuracy=0., best_model=False):
    """Save model checkpoint

    :param configs: Configuration settings
    :param model: Model instance
    :param optimizer: Optimizer instance
    :param amp_scaler: Automatic Mixed Precision (AMP) scaler
    :param save_model_path: Path to save the model
    :param epoch_id: Current epoch number
    :param accuracy: Current evaluation accuracy
    :param best_model: Whether this is the best model
    """
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    # Determine save path
    save_feature_method = configs.preprocess_conf.feature_method
    if configs.preprocess_conf.get('use_hf_model', False):
        save_feature_method = save_feature_method.rstrip('/')
        save_feature_method = os.path.basename(save_feature_method)

    if best_model:
        model_path = os.path.join(save_model_path,
                                  f'{configs.model_conf.model}_{save_feature_method}',
                                  'best_model')
    else:
        model_path = os.path.join(save_model_path,
                                  f'{configs.model_conf.model}_{save_feature_method}',
                                  f'epoch_{epoch_id}')

    os.makedirs(model_path, exist_ok=True)

    # Save model and optimizer states
    torch.save(optimizer.state_dict(), os.path.join(model_path, 'optimizer.pth'))
    torch.save(state_dict, os.path.join(model_path, 'model.pth'))

    # Save AMP scaler if available
    if amp_scaler is not None:
        torch.save(amp_scaler.state_dict(), os.path.join(model_path, 'scaler.pth'))

    # Save additional metadata
    with open(os.path.join(model_path, 'model.state'), 'w', encoding='utf-8') as f:
        data = {
            "last_epoch": epoch_id,
            "accuracy": accuracy,
            "version": __version__,
            "model": configs.model_conf.model,
            "feature_method": save_feature_method
        }
        f.write(json.dumps(data, indent=4, ensure_ascii=False))

    if not best_model:
        # Update the "last_model" checkpoint
        last_model_path = os.path.join(save_model_path,
                                       f'{configs.model_conf.model}_{save_feature_method}',
                                       'last_model')
        shutil.rmtree(last_model_path, ignore_errors=True)
        shutil.copytree(model_path, last_model_path)

        # Remove old checkpoints (older than 3 epochs ago)
        old_model_path = os.path.join(save_model_path,
                                      f'{configs.model_conf.model}_{save_feature_method}',
                                      f'epoch_{epoch_id - 3}')
        if os.path.exists(old_model_path):
            shutil.rmtree(old_model_path)

    logger.info('Model saved at: {}'.format(model_path))
