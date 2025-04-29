import argparse
import functools

from mser.trainer import MSERTrainer
from mser.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',              str,    'configs/bi_lstm.yml',      'Configuration file')
add_arg('data_augment_configs', str,    'configs/augmentation.yml', 'Data augmentation configuration file')
add_arg("local_rank",           int,    0,                          'Parameter required for multi-GPU training')
add_arg("use_gpu",              bool,   True,                       'Whether to use GPU for training')
add_arg('save_model_path',      str,    'models/',                  'Path to save the trained model')
add_arg('log_dir',              str,    'log/',                     'Path to save VisualDL log files')
add_arg('resume_model',         str,    None,                       'Path to resume training; if None, do not use a resume model')
add_arg('pretrained_model',     str,    None,                       'Path to pretrained model; if None, do not use a pretrained model')
add_arg('overwrites',           str,    None,    'Override parameters in the config file, e.g., "train_conf.max_epoch=100"; multiple overrides separated by commas')
args = parser.parse_args()
print_arguments(args=args)

# Get the trainer
trainer = MSERTrainer(configs=args.configs,
                      use_gpu=args.use_gpu,
                      data_augment_configs=args.data_augment_configs,
                      overwrites=args.overwrites)

# Start training
trainer.train(save_model_path=args.save_model_path,
              log_dir=args.log_dir,
              resume_model=args.resume_model,
              pretrained_model=args.pretrained_model)
