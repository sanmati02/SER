import argparse
import functools

from mser.trainer import MSERTrainer
from mser.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'configs/bi_lstm.yml',      'Configuration file')
add_arg('save_dir',         str,    'dataset/features',         'Path to save extracted features')
add_arg('max_duration',     int,     100,                       'Maximum duration for feature extraction (in seconds)')
args = parser.parse_args()
print_arguments(args=args)

# Get the trainer
trainer = MSERTrainer(configs=args.configs)

# Extract features and save them
trainer.extract_features(save_dir=args.save_dir, max_duration=args.max_duration)

# Generate normalization file
trainer.get_standard_file(max_duration=args.max_duration)
