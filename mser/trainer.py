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
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns, time, os, json
import os, time, json, numpy as np, torch
import pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

import json, pathlib
MAPPING_PATH = pathlib.Path('dataset/npy_to_wav.json')
_npy2wav = json.load(MAPPING_PATH.open()) if MAPPING_PATH.exists() else {}


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

        self.epoch_train_loss = []
        self.epoch_train_acc = []
        self.epoch_eval_loss = []
        self.epoch_eval_acc = []

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

    
    
    def _save_confusion(cm, labels, save_dir):
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted"); plt.ylabel("True")
        os.makedirs(save_dir, exist_ok=True)
        fp = os.path.join(save_dir, f"confusion_{int(time.time())}.png")
        plt.tight_layout(); plt.savefig(fp); plt.close()
        print(f"✓ confusion matrix saved → {fp}")


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
            feature = test_dataset[i][0]
            if feature.ndim == 2:  # [T, F]
                data.extend(feature)  # Add each [F] timestep
            elif feature.ndim == 1:
                data.append(feature)  # Already [F]
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
                    feature, label, *_ = test_dataset[i]
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
            # summary(self.model, input_size=(1, self.test_dataset.audio_featurizer.feature_dim))
            example_input = self.test_dataset[0][0]  # shape: [T, F] for sequence models

            if example_input.ndim != 2:
                raise ValueError(f"Expected input of shape [T, F], got {example_input.shape}")
    
            T, F = example_input.shape
            input_tensor = torch.tensor(example_input, dtype=torch.float32).unsqueeze(0).to(self.device)  # [1, T, F]
    
            print(f"[Summary] Using real input shape: {input_tensor.shape}")
            summary(self.model, input_data=input_tensor)

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
        """Train for one epoch"""
        epoch_total_loss = 0.0
        epoch_total_correct = 0
        epoch_total_samples = 0
        train_times, accuracies, loss_sum = [], [], []
        start = time.time()
    
        for batch_id, batch in enumerate(self.train_loader):
            if len(batch)==3: 
                features, label, input_lens_ratio = batch
            else: 
                features, label, input_lens_ratio, _ = batch
            if self.stop_train: break
            if nranks > 1:
                features = features.to(local_rank)
                label = label.to(local_rank).long()
            else:
                features = features.to(self.device)
                label = label.to(self.device).long()
    
            with torch.autocast('cuda', enabled=self.configs.train_conf.enable_amp):
                output = self.model(features)

            logits = output[0] if isinstance(output, tuple) else output


            los = self.loss(logits, label)
    
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
    
            batch_size = label.size(0)
            epoch_total_loss += los.item() * batch_size
            epoch_total_correct += acc * batch_size
            epoch_total_samples += batch_size
    
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
    
        avg_loss = epoch_total_loss / epoch_total_samples
        avg_acc = epoch_total_correct / epoch_total_samples
        return avg_loss, avg_acc


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

        epochs_since_improvement = 0
        early_stopping_patience = 20

        for epoch_id in range(last_epoch, self.configs.train_conf.max_epoch):
            if self.stop_train: break
            epoch_id += 1
            start_epoch = time.time()
        
            avg_loss, avg_acc = self.__train_epoch(epoch_id=epoch_id, local_rank=local_rank, writer=writer, nranks=nranks)
        
            if local_rank == 0:
                self.epoch_train_loss.append(avg_loss)
                self.epoch_train_acc.append(avg_acc)
                writer.add_scalar('Train/Epoch_Loss', avg_loss, epoch_id)
                writer.add_scalar('Train/Epoch_Accuracy', avg_acc, epoch_id)
        
                if self.stop_eval: continue
                logger.info('=' * 70)
                self.eval_loss, self.eval_acc = self.evaluate()
                logger.info('Test epoch: {}, time/epoch: {}, loss: {:.5f}, accuracy: {:.5f}'.format(
                    epoch_id, str(timedelta(seconds=(time.time() - start_epoch))), self.eval_loss, self.eval_acc))
                logger.info('=' * 70)
                self.epoch_eval_loss.append(self.eval_loss)
                self.epoch_eval_acc.append(self.eval_acc)
        
                writer.add_scalar('Test/Accuracy', self.eval_acc, self.test_log_step)
                writer.add_scalar('Test/Loss', self.eval_loss, self.test_log_step)
                self.test_log_step += 1
                self.model.train()

                if self.eval_acc >= best_acc:
                    best_acc = self.eval_acc
                    epochs_since_improvement = 0
                    save_checkpoint(configs=self.configs, model=self.model, optimizer=self.optimizer,
                                    amp_scaler=self.amp_scaler, save_model_path=save_model_path, epoch_id=epoch_id,
                                    accuracy=self.eval_acc, best_model=True)
                else:
                    epochs_since_improvement += 1
        
                save_checkpoint(configs=self.configs, model=self.model, optimizer=self.optimizer,
                                amp_scaler=self.amp_scaler, save_model_path=save_model_path, epoch_id=epoch_id,
                                accuracy=self.eval_acc)

                if epochs_since_improvement >= early_stopping_patience and epoch_id > 10:
                    logger.info(f"Early stopping triggered after {early_stopping_patience} epochs with no improvement.")
                    break

    
        
        # === Compose filename ===
        model_name = self.configs.get("model", {}).get("name", "model")
        max_epoch = self.configs.train_conf.max_epoch
        trial_name = getattr(self, "trial_name", "default")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename_prefix = f"{model_name}_ep{max_epoch}_{trial_name}_{timestamp}"
        loss_curve_path = f"{filename_prefix}_loss_curve.png"
        acc_curve_path = f"{filename_prefix}_accuracy_curve.png"
        
        # === Plotting ===
        epochs = list(range(1, len(self.epoch_train_loss) + 1))
        
        plt.figure()
        plt.plot(epochs, self.epoch_train_loss, label='Train Loss')
        plt.plot(epochs, self.epoch_eval_loss, label='Eval Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()
        plt.savefig(loss_curve_path)
        
        plt.figure()
        plt.plot(epochs, self.epoch_train_acc, label='Train Accuracy')
        plt.plot(epochs, self.epoch_eval_acc, label='Eval Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curve')
        plt.legend()
        plt.savefig(acc_curve_path)
        
        logger.info(f"Saved loss curve to {loss_curve_path}")
        logger.info(f"Saved accuracy curve to {acc_curve_path}")



    def summarize_attention_all_emotions(self, attention_outputs, predictions, labels, class_labels, save_dir="attention_analysis"):
        os.makedirs(save_dir, exist_ok=True)
    
        # Organize weights by emotion and correctness
        correct_weights = {label: [] for label in class_labels}
        incorrect_weights = {label: [] for label in class_labels}
        all_weights = {label: [] for label in class_labels}
    
        for attn_output, pred, true in zip(attention_outputs, predictions, labels):
            #weights = attn_output.detach().cpu().numpy()  # shape [T]
            weights = attn_output["weights"]
            pred = attn_output["pred"]
            true = attn_output["true"]

            max_len = max(len(w) for w in [weights])
            if len(weights) < max_len:
                weights = np.pad(weights, (0, max_len - len(weights)), mode='constant')
            emotion = class_labels[true]
            all_weights[emotion].append(weights)
            if pred == true:
                correct_weights[emotion].append(weights)
            else:
                incorrect_weights[emotion].append(weights)
    
        # === Figure 1: Mean ± Std per emotion ===
        fig, axes = plt.subplots(1, len(class_labels), figsize=(28, 4), sharey=True)
        for idx, emotion in enumerate(class_labels):
            ax = axes[idx]
            if correct_weights[emotion]:
                corr_raw = correct_weights[emotion]
                max_len = max(len(w) for w in corr_raw)
                corr_padded = [np.pad(w, (0, max_len - len(w)), mode='constant') for w in corr_raw]
                corr = np.stack(corr_padded)
                mean_corr = corr.mean(axis=0)
                std_corr = corr.std(axis=0)
                ax.plot(mean_corr, label='Correct', color='blue')
                ax.fill_between(np.arange(len(mean_corr)), mean_corr - std_corr, mean_corr + std_corr,
                                color='blue', alpha=0.2)
            if incorrect_weights[emotion]:
                incorr_raw = incorrect_weights[emotion]
                max_len = max(len(w) for w in incorr_raw)
                incorr_padded = [np.pad(w, (0, max_len - len(w)), mode='constant') for w in incorr_raw]
                incorr = np.stack(incorr_padded)
                mean_incorr = incorr.mean(axis=0)
                std_incorr = incorr.std(axis=0)
                ax.plot(mean_incorr, label='Incorrect', color='red', linestyle='--')
                ax.fill_between(np.arange(len(mean_incorr)), mean_incorr - std_incorr, mean_incorr + std_incorr,
                                color='red', alpha=0.2)
            ax.set_title(emotion)
            ax.set_xlabel("Timestep")
        axes[0].set_ylabel("Attention Weight")
        fig.suptitle("Mean ± Std Attention per Emotion (Correct vs Incorrect)", fontsize=14)
        fig.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "attention_mean_std_all_emotions.png"))
        plt.close()
    
        import matplotlib.gridspec as gridspec

        
        fig = plt.figure(figsize=(10, 10))  # Reduce width to make plots less wide
        gs = gridspec.GridSpec(4, 2, figure=fig)  # 4 rows, 2 columns
        emotion_axes = []
        
        for idx, emotion in enumerate(class_labels):
            ax = fig.add_subplot(gs[idx])
            heat_raw = all_weights[emotion]
            max_len = max(len(w) for w in heat_raw)
            heat_padded = [np.pad(w, (0, max_len - len(w)), mode='constant') for w in heat_raw]
            sns.heatmap(np.stack(heat_padded), ax=ax, cmap='Blues', cbar=False)
        
            ax.set_title(emotion, fontsize=10)
            ax.set_xlabel("Timestep", fontsize=8)
            ax.set_ylabel("Sample Index", fontsize=8)
            emotion_axes.append(ax)
        
        # Hide extra subplot if fewer than 8 emotions
        for idx in range(len(class_labels), 8):
            fig.add_subplot(gs[idx]).axis('off')
        
        fig.suptitle("Attention Weight Heatmaps per Emotion", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(save_dir, "attention_heatmaps_all_emotions.png"))
        plt.close()



    
        return f"Saved mean/std and heatmap plots to: {save_dir}"
    
    
        
    def evaluate(self,
                 resume_model: str | None = None,
                 save_dir:     str | None = None):
        """
        Run evaluation and (optionally) write artefacts to disk.
    
        Parameters
        ----------
        resume_model : str | None
            Path to a checkpoint directory or *.pth file to evaluate.
        save_dir : str | None
            Directory where confusion-matrix PNG, class report (JSON) and
            mis-classification list (CSV) will be written.  If None, no files
            are saved.
    
        Returns
        -------
        loss : float   Average cross-entropy loss over the test set.
        acc  : float   Average accuracy  over the test set.
        """
    
        
        if self.test_loader is None:
            self.__setup_dataloader()
        if self.model is None:
            self.__setup_model(
                input_size=self.test_dataset.audio_featurizer.feature_dim
            )
    
        if resume_model is not None:
            ckpt = resume_model
            if os.path.isdir(ckpt):
                ckpt = os.path.join(ckpt, "model.pth")
            assert os.path.exists(ckpt), f"{ckpt} does not exist!"
            self.model.load_state_dict(torch.load(ckpt, weights_only=False),
                                       strict=False)
            logger.info(f"✓ loaded checkpoint {ckpt}")
    
        self.model.eval()
        eval_model = self.model.module if isinstance(
            self.model, torch.nn.parallel.DistributedDataParallel) else self.model
    
       
        losses, accs, all_lbls, all_preds, wrong = [], [], [], [], []
 
        def _parse_meta(path: str) -> dict:
            """
            Return a meta-data dict for common SER corpora.
            Works for:
              • RAVDESS   03-01-02-01-01-01-12.wav
             
            Anything else → empty dict.
            """
            name = os.path.basename(path)
            stem = os.path.splitext(name)[0]
        
            if '-' in stem and len(stem.split('-')) == 7:
                t = stem.split('-')
                return dict(
                    corpus    = 'RAVDESS',
                    modality  = int(t[0]),
                    channel   = int(t[1]),
                    emotion   = int(t[2]),
                    intensity = int(t[3]),
                    statement = int(t[4]),
                    repetition= int(t[5]),
                    actor     = int(t[6]),
                )
        
            
            return {}

    
        with torch.no_grad():
            
            attention_outputs = []

            for batch in tqdm(self.test_loader, desc="Eval"):
                # Accept either 3- or 4-tuple batches
                features, label, *rest = batch                    # rest = [len] OR [len, path]
                input_lens_ratio = rest[0]
                paths = rest[1] if len(rest) == 2 else None       # None if dataloader doesn't send paths
                features = features.to(self.device)
                label    = label.to(self.device).long()
    
                output = eval_model(features)
                
                # Only unpack if output is a tuple
                if isinstance(output, tuple):
                    attn_weights = output[1]  # Save for later
                    output = output[0]
                
                loss = self.loss(output, label)
                losses.append(loss.item())
                acc = accuracy(output, label)
                accs.append(acc)
                
                probs = output.softmax(dim=1).cpu().numpy()
                preds = probs.argmax(axis=1)
                lbls = label.cpu().numpy()
                
                all_preds.extend(preds.tolist())
                all_lbls.extend(lbls.tolist())
                
                # collect attention *after* preds are computed
                if self.configs.model_conf.model in ['LSTMAdditiveAttention', 'StackedLSTMAdditiveAttention']:
                    attention_outputs.extend([
                        {
                            "weights": w.detach().cpu().numpy(),
                            "true": int(t),
                            "pred": int(p)
                        }
                        for w, t, p in zip(attn_weights, lbls, preds)
                    ])
    
                # collect mis-predictions if we have file paths
                if paths is not None:
                    for p, t, path in zip(preds, lbls, paths):
                        if p != t:
                            wav_path = _npy2wav.get(path, path)   # fall back to itself if missing
                            wrong.append({
                                'path': path,      # the .npy (kept for debugging)
                                'wav':  wav_path,  # original RAVDESS name if available
                                'true': int(t),
                                'pred': int(p),
                                **_parse_meta(wav_path)   # actor, intensity, etc.
                            })


        if self.configs.model_conf.model == 'LSTMAdditiveAttention' or self.configs.model_conf.model == 'StackedLSTMAdditiveAttention':
            self.summarize_attention_all_emotions(
                attention_outputs=attention_outputs,
                predictions=all_preds,
                labels=all_lbls,
                class_labels=["Neutral", "Happy", "Sad", "Angry", "Fearful", "Disgusted", "Surprised"],
                save_dir="attention_plots"
            )


    
       
        avg_loss = float(np.mean(losses)) if losses else -1
        avg_acc  = float(np.mean(accs))   if accs  else -1
    
      
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
    
            # 4-a Confusion matrix PNG
            cm = confusion_matrix(all_lbls, all_preds, labels=list(range(len(self.class_labels))))
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_percent, annot=True, fmt=".2f", cmap="Reds",
                        xticklabels=self.class_labels,
                        yticklabels=self.class_labels,
                        cbar_kws={'label': 'Percentage (%)'}, vmin=0, vmax=100)
            plt.title("Confusion Matrix Heatmap (Percentages)")
            plt.xlabel("Predicted Emotion")
            plt.ylabel("True Emotion")
            plt.tight_layout()
            
            os.makedirs(save_dir, exist_ok=True)
            fname = os.path.join(save_dir, f"confusion_{self.configs.model_conf.model}.png")
            plt.savefig(fname)
            plt.close()
            logger.info(f"✓ confusion matrix (percentage) saved → {fname}")
    
            # 4-b Per-class precision / recall / F1
            report = classification_report(all_lbls, all_preds,
                                           target_names=self.class_labels,
                                           output_dict=True)
            with open(os.path.join(save_dir, f"class_report_{self.configs.model_conf.model}.json"), "w") as fp:
                json.dump(report, fp, indent=2)
    
            # 4-c CSV of mistakes
            if wrong:
                pd.DataFrame(wrong).to_csv(
                    os.path.join(save_dir, f"misclassified_{self.configs.model_conf.model}.csv"), index=False)
                logger.info(f"✓ {len(wrong)} mis-classified samples written")
    
        # reset to train mode for further training
        self.model.train()
        return avg_loss, avg_acc

    def export(self, save_model_path='models/', resume_model='models/BiLSTM_CustomFeature/best_model/'):
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
