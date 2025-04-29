import os
import platform
import sys
import time
from datetime import timedelta

import joblib
import numpy as np
import torch
import torch.distributed as dist
import yaml
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torchinfo import summary
from tqdm import tqdm
from visualdl import LogWriter

from loguru import logger
from mser.data_utils.collate_fn import collate_fn
from mser.data_utils.featurizer import AudioFeaturizer
from mser.data_utils.reader import CustomDataset
from mser.metric.metrics import accuracy
from mser.models import build_model
from mser.optimizer import build_optimizer, build_lr_scheduler
from mser.utils.checkpoint import load_pretrained, load_checkpoint, save_checkpoint
from mser.utils.utils import dict_to_object, plot_confusion_matrix, print_arguments, convert_string_based_on_type


class MSERTrainer(object):
    def __init__(self,
                 configs,
                 use_gpu=True,
                 data_augment_configs=None,
                 num_class=None,
                 overwrites=None,
                 log_level="info"):
        """Speech Emotion Recognition training utility class

        :param configs: Config file path or model name. If a model name is given, the default config file is used.
        :param use_gpu: Whether to train the model using GPU
        :param data_augment_configs: Data augmentation config dictionary or path to it
        :param num_class: Number of classes, corresponds to model_conf.model_args.num_class in the config
        :param overwrites: Parameters to override in the config file, e.g., "train_conf.max_epoch=100", separated by commas
        :param log_level: Logging level, options: "debug", "info", "warning", "error"
        """
        if use_gpu:
            assert (torch.cuda.is_available()), 'GPU not available'

            self.device = torch.device("cuda")
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            self.device = torch.device("cpu")
        self.use_gpu = use_gpu
        self.log_level = log_level.upper()
        logger.remove()
        logger.add(sink=sys.stdout, level=self.log_level)

        if isinstance(configs, str):

            absolute_path = os.path.dirname(__file__)
            config_path = os.path.join(absolute_path, f"configs/{configs}.yml")
            configs = config_path if os.path.exists(config_path) else configs
            with open(configs, 'r', encoding='utf-8') as f:
                configs = yaml.load(f.read(), Loader=yaml.FullLoader)
        self.configs = dict_to_object(configs)
        if num_class is not None:
            self.configs.model_conf.model_args.num_class = num_class

        if overwrites:
            overwrites = overwrites.split(",")
            for overwrite in overwrites:
                keys, value = overwrite.strip().split("=")
                attrs = keys.split('.')
                current_level = self.configs
                for attr in attrs[:-1]:
                    current_level = getattr(current_level, attr)
                before_value = getattr(current_level, attrs[-1])
                setattr(current_level, attrs[-1], convert_string_based_on_type(before_value, value))

        print_arguments(configs=self.configs)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.audio_featurizer = None
        self.train_dataset = None
        self.train_loader = None
        self.test_dataset = None
        self.test_loader = None
        self.amp_scaler = None

        if isinstance(data_augment_configs, str):
            with open(data_augment_configs, 'r', encoding='utf-8') as f:
                data_augment_configs = yaml.load(f.read(), Loader=yaml.FullLoader)
            print_arguments(configs=data_augment_configs, title='数据增强配置')
        self.data_augment_configs = dict_to_object(data_augment_configs)

        with open(self.configs.dataset_conf.label_list_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.class_labels = [l.replace('\n', '') for l in lines]
        if platform.system().lower() == 'windows':
            self.configs.dataset_conf.dataLoader.num_workers = 0
            logger.warning('Windows does not support multi-threaded data loading. Using single-threaded loading.')

        if self.configs.preprocess_conf.feature_method == 'Emotion2Vec':
            self.configs.dataset_conf.dataLoader.num_workers = 0
            logger.warning('Emotion2Vec feature extraction does not support multithreading. Switching to single-threaded.')
        self.max_step, self.train_step = None, None
        self.train_loss, self.train_acc = None, None
        self.train_eta_sec = None
        self.eval_loss, self.eval_acc = None, None
        self.test_log_step, self.train_log_step = 0, 0
        self.stop_train, self.stop_eval = False, False

    def __setup_dataloader(self, is_train=False):
        self.audio_featurizer = AudioFeaturizer(feature_method=self.configs.preprocess_conf.feature_method,
                                                method_args=self.configs.preprocess_conf.get('method_args', {}))

        dataset_args = self.configs.dataset_conf.get('dataset', {})
        data_loader_args = self.configs.dataset_conf.get('dataLoader', {})
        if is_train:
            self.train_dataset = CustomDataset(data_list_path=self.configs.dataset_conf.train_list,
                                               audio_featurizer=self.audio_featurizer,
                                               aug_conf=self.data_augment_configs,
                                               mode='train',
                                               **dataset_args)

            train_sampler = RandomSampler(self.train_dataset)
            if torch.cuda.device_count() > 1:

                train_sampler = DistributedSampler(dataset=self.train_dataset)
            self.train_loader = DataLoader(dataset=self.train_dataset,
                                           collate_fn=collate_fn,
                                           sampler=train_sampler,
                                           **data_loader_args)

        data_loader_args.drop_last = False
        dataset_args.max_duration = self.configs.dataset_conf.eval_conf.max_duration
        data_loader_args.batch_size = self.configs.dataset_conf.eval_conf.batch_size
        self.test_dataset = CustomDataset(data_list_path=self.configs.dataset_conf.test_list,
                                          audio_featurizer=self.audio_featurizer,
                                          mode='eval',
                                          **dataset_args)
        self.test_loader = DataLoader(dataset=self.test_dataset,
                                      collate_fn=collate_fn,
                                      shuffle=False,
                                      **data_loader_args)


    def get_standard_file(self, max_duration=100):
        self.audio_featurizer = AudioFeaturizer(feature_method=self.configs.preprocess_conf.feature_method,
                                                method_args=self.configs.preprocess_conf.get('method_args', {}))

        dataset_args = self.configs.dataset_conf.get('dataset', {})
        dataset_args.max_duration = max_duration
        test_dataset = CustomDataset(data_list_path=self.configs.dataset_conf.train_list,
                                     audio_featurizer=self.audio_featurizer,
                                     mode='create_data',
                                     **dataset_args)
        data = []
        for i in tqdm(range(len(test_dataset))):
            feature, _ = test_dataset[i]
            data.append(feature)
        scaler = StandardScaler().fit(data)
        joblib.dump(scaler, self.configs.dataset_conf.dataset.scaler_path)
        logger.info(f'Normalization file saved to: {self.configs.dataset_conf.dataset.scaler_path}')


    def extract_features(self, save_dir='dataset/features', max_duration=100):
        """Extract features and save to file

        :param save_dir: Save directory
        :param max_duration: Maximum duration
        """

        self.audio_featurizer = AudioFeaturizer(feature_method=self.configs.preprocess_conf.feature_method,
                                                method_args=self.configs.preprocess_conf.get('method_args', {}))
        for j, data_list in enumerate([self.configs.dataset_conf.train_list, self.configs.dataset_conf.test_list]):

            dataset_args = self.configs.dataset_conf.get('dataset', {})
            dataset_args.max_duration = max_duration
            test_dataset = CustomDataset(data_list_path=data_list,
                                         audio_featurizer=self.audio_featurizer,
                                         mode='extract_feature',
                                         **dataset_args)
            save_data_list = data_list.replace('.txt', '_features.txt')
            if j == 0:
                self.configs.dataset_conf.train_list = save_data_list
            elif j == 1:
                self.configs.dataset_conf.test_list = save_data_list
            save_data_list = data_list.replace('.txt', '_features.txt')
            with open(save_data_list, 'w', encoding='utf-8') as f:
                for i in tqdm(range(len(test_dataset))):
                    feature, label = test_dataset[i]
                    label = int(label)
                    save_path = os.path.join(save_dir, str(label), f'{int(time.time() * 1000)}.npy')
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    np.save(save_path, feature)
                    f.write(f'{save_path}\t{label}\n')
            logger.info(f'Data in {data_list} extracted. New list saved as: {save_data_list}')


    def __setup_model(self, input_size, is_train=False):
        """Setup the model

        :param input_size: Input feature size for the model
        :param is_train: Whether to prepare the model for training
        """

        if self.configs.model_conf.model_args.get('num_class', None) is None:
            self.configs.model_conf.model_args.num_class = len(self.class_labels)

        self.model = build_model(input_size=input_size, configs=self.configs)
        self.model.to(self.device)
        if self.log_level == "DEBUG" or self.log_level == "INFO":
            summary(self.model, input_size=(1, self.test_dataset.audio_featurizer.feature_dim))

        label_smoothing = self.configs.train_conf.get('label_smoothing', 0.0)
        self.loss = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        if is_train:
            if self.configs.train_conf.enable_amp:
                self.amp_scaler = torch.cuda.amp.GradScaler(init_scale=1024)

            self.optimizer = build_optimizer(params=self.model.parameters(), configs=self.configs)

            self.scheduler = build_lr_scheduler(optimizer=self.optimizer, step_per_epoch=len(self.train_loader),
                                                configs=self.configs)
        if self.configs.train_conf.use_compile and torch.__version__ >= "2" and platform.system().lower() != 'windows':
            self.model = torch.compile(self.model, mode="reduce-overhead")

    def __train_epoch(self, epoch_id, local_rank, writer, nranks=0):
        """Train for one epoch

        :param epoch_id: Current epoch ID
        :param local_rank: Local GPU ID
        :param writer: VisualDL writer object
        :param nranks: Number of GPUs used
        """

        train_times, accuracies, loss_sum = [], [], []
        start = time.time()
        for batch_id, (features, label, input_lens_ratio) in enumerate(self.train_loader):
            if self.stop_train: break
            if nranks > 1:
                features = features.to(local_rank)
                label = label.to(local_rank).long()
            else:
                features = features.to(self.device)
                label = label.to(self.device).long()

            with torch.autocast('cuda', enabled=self.configs.train_conf.enable_amp):
                output = self.model(features)

            los = self.loss(output, label)

            if self.configs.train_conf.enable_amp:

                scaled = self.amp_scaler.scale(los)
                scaled.backward()
            else:
                los.backward()

            if self.configs.train_conf.enable_amp:
                self.amp_scaler.unscale_(self.optimizer)
                self.amp_scaler.step(self.optimizer)
                self.amp_scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()


            acc = accuracy(output, label)
            accuracies.append(acc)
            loss_sum.append(los.data.cpu().numpy())
            train_times.append((time.time() - start) * 1000)
            self.train_step += 1


            if batch_id % self.configs.train_conf.log_interval == 0 and local_rank == 0:

                train_speed = self.configs.dataset_conf.dataLoader.batch_size / (
                        sum(train_times) / len(train_times) / 1000)

                self.train_eta_sec = (sum(train_times) / len(train_times)) * (self.max_step - self.train_step) / 1000
                eta_str = str(timedelta(seconds=int(self.train_eta_sec)))
                self.train_loss = sum(loss_sum) / len(loss_sum)
                self.train_acc = sum(accuracies) / len(accuracies)
                logger.info(f'Train epoch: [{epoch_id}/{self.configs.train_conf.max_epoch}], '
                            f'batch: [{batch_id}/{len(self.train_loader)}], '
                            f'loss: {self.train_loss:.5f}, accuracy: {self.train_acc:.5f}, '
                            f'learning rate: {self.scheduler.get_last_lr()[0]:>.8f}, '
                            f'speed: {train_speed:.2f} data/sec, eta: {eta_str}')
                writer.add_scalar('Train/Loss', self.train_loss, self.train_log_step)
                writer.add_scalar('Train/Accuracy', self.train_acc, self.train_log_step)
                writer.add_scalar('Train/lr', self.scheduler.get_last_lr()[0], self.train_log_step)
                train_times, accuracies, loss_sum = [], [], []
                self.train_log_step += 1
            start = time.time()
            self.scheduler.step()

    def train(self,
              save_model_path='models/',
              log_dir='log/',
              max_epoch=None,
              resume_model=None,
              pretrained_model=None):
        """
        Train the model

        :param save_model_path: Path to save trained models
        :param log_dir: Path to save VisualDL logs
        :param max_epoch: Maximum training epochs
        :param resume_model: Path to checkpoint to resume training
        :param pretrained_model: Path to pretrained model
        """
        nranks = torch.cuda.device_count()
        local_rank = 0
        writer = None
        if local_rank == 0:

            writer = LogWriter(logdir=log_dir)

        if nranks > 1 and self.use_gpu:

            dist.init_process_group(backend='nccl')
            local_rank = int(os.environ["LOCAL_RANK"])


        self.__setup_dataloader(is_train=True)

        self.__setup_model(input_size=self.test_dataset.audio_featurizer.feature_dim, is_train=True)

        self.model = load_pretrained(model=self.model, pretrained_model=pretrained_model)

        self.model, self.optimizer, self.amp_scaler, self.scheduler, last_epoch, best_acc = \
            load_checkpoint(configs=self.configs, model=self.model, optimizer=self.optimizer,
                            amp_scaler=self.amp_scaler, scheduler=self.scheduler, step_epoch=len(self.train_loader),
                            save_model_path=save_model_path, resume_model=resume_model)

        if nranks > 1 and self.use_gpu:
            self.model.to(local_rank)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank])
        logger.info('Training data size: {}'.format(len(self.train_dataset)))

        self.train_loss, self.train_acc = None, None
        self.eval_loss, self.eval_acc = None, None
        self.test_log_step, self.train_log_step = 0, 0
        if local_rank == 0:
            writer.add_scalar('Train/lr', self.scheduler.get_last_lr()[0], last_epoch)
        if max_epoch is not None:
            self.configs.train_conf.max_epoch = max_epoch

        self.max_step = len(self.train_loader) * self.configs.train_conf.max_epoch
        self.train_step = max(last_epoch, 0) * len(self.train_loader)

        for epoch_id in range(last_epoch, self.configs.train_conf.max_epoch):
            if self.stop_train: break
            epoch_id += 1
            start_epoch = time.time()

            self.__train_epoch(epoch_id=epoch_id, local_rank=local_rank, writer=writer, nranks=nranks)

            if local_rank == 0:
                if self.stop_eval: continue
                logger.info('=' * 70)
                self.eval_loss, self.eval_acc = self.evaluate()
                logger.info('Test epoch: {}, time/epoch: {}, loss: {:.5f}, accuracy: {:.5f}'.format(
                    epoch_id, str(timedelta(seconds=(time.time() - start_epoch))), self.eval_loss, self.eval_acc))
                logger.info('=' * 70)
                writer.add_scalar('Test/Accuracy', self.eval_acc, self.test_log_step)
                writer.add_scalar('Test/Loss', self.eval_loss, self.test_log_step)
                self.test_log_step += 1
                self.model.train()

                if self.eval_acc >= best_acc:
                    best_acc = self.eval_acc
                    save_checkpoint(configs=self.configs, model=self.model, optimizer=self.optimizer,
                                    amp_scaler=self.amp_scaler, save_model_path=save_model_path, epoch_id=epoch_id,
                                    accuracy=self.eval_acc, best_model=True)

                save_checkpoint(configs=self.configs, model=self.model, optimizer=self.optimizer,
                                amp_scaler=self.amp_scaler, save_model_path=save_model_path, epoch_id=epoch_id,
                                accuracy=self.eval_acc)

    def evaluate(self, resume_model=None, save_matrix_path=None):
        """
        Evaluate the model

        :param resume_model: Model to use for evaluation
        :param save_matrix_path: Path to save the confusion matrix
        :return: Evaluation loss and accuracy
        """
        if self.test_loader is None:
            self.__setup_dataloader()
        if self.model is None:
            self.__setup_model(input_size=self.test_dataset.audio_featurizer.feature_dim)
        if resume_model is not None:
            if os.path.isdir(resume_model):
                resume_model = os.path.join(resume_model, 'model.pth')
            assert os.path.exists(resume_model), f"{resume_model} Model does not exist!"
            model_state_dict = torch.load(resume_model, weights_only=False)
            self.model.load_state_dict(model_state_dict)
            logger.info(f'Successfully loaded model: {resume_model}')
        self.model.eval()
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            eval_model = self.model.module
        else:
            eval_model = self.model

        accuracies, losses, preds, labels = [], [], [], []
        with torch.no_grad():
            for batch_id, (features, label, input_lens_ratio) in enumerate(tqdm(self.test_loader, desc='Perform Evaluation')):
                if self.stop_eval: break
                features = features.to(self.device)
                label = label.to(self.device).long()
                output = eval_model(features)
                los = self.loss(output, label)

                acc = accuracy(output, label)
                accuracies.append(acc)

                label = label.data.cpu().numpy()
                output = output.data.cpu().numpy()
                pred = np.argmax(output, axis=1)
                preds.extend(pred.tolist())

                labels.extend(label.tolist())
                los1 = los.data.cpu().numpy()
                if not np.isnan(los1):
                    losses.append(los1)
        loss = float(sum(losses) / len(losses)) if len(losses) > 0 else -1
        acc = float(sum(accuracies) / len(accuracies)) if len(accuracies) > 0 else -1

        if save_matrix_path is not None:
            try:
                cm = confusion_matrix(labels, preds)
                plot_confusion_matrix(cm=cm, save_path=os.path.join(save_matrix_path, f'{int(time.time())}.png'),
                                      class_labels=self.class_labels)
            except Exception as e:
                logger.error(f'Failed to save confusion matrix: {e}')
        self.model.train()
        return loss, acc

    def export(self, save_model_path='models/', resume_model='models/BiLSTM_Emotion2Vec/best_model/'):
        """
        Export the inference model

        :param save_model_path: Path to save the exported model
        :param resume_model: Path to the model to convert
        """
        self.__setup_model(input_size=self.test_dataset.audio_featurizer.feature_dim)

        if os.path.isdir(resume_model):
            resume_model = os.path.join(resume_model, 'model.pth')
        assert os.path.exists(resume_model), f"{resume_model} 模型不存在！"
        model_state_dict = torch.load(resume_model)
        self.model.load_state_dict(model_state_dict)
        logger.info("Inference model saved to: {}".format(infer_model_path))
        self.model.eval()

        infer_model = self.model.export()
        infer_model_path = os.path.join(save_model_path,
                                        f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                        'inference.pth')
        os.makedirs(os.path.dirname(infer_model_path), exist_ok=True)
        torch.jit.save(infer_model, infer_model_path)
        logger.info("Inference model saved to: {}".format(infer_model_path))
