import argparse
import functools
import time

from mser.trainer import MSERTrainer
from mser.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,   'configs/bi_lstm.yml',    "Configuration file")
add_arg("use_gpu",          bool,  True,                     "Whether to use GPU for model evaluation")
add_arg('save_matrix_path', str,   'analysis/',         "Path to save the confusion matrix")
add_arg('resume_model',     str,   'models/BiLSTM_CustomFeature/best_model/',  "Path to the pretrained model")
add_arg('overwrites',       str,    None,    'Override parameters in the config file, e.g., "train_conf.max_epoch=100"; use commas to separate multiple overrides')
args = parser.parse_args()
print_arguments(args=args)

# Get the trainer
trainer = MSERTrainer(configs=args.configs, use_gpu=args.use_gpu, overwrites=args.overwrites)

# Start evaluation
start = time.time()
loss, accuracy = trainer.evaluate(resume_model=args.resume_model,
                                  save_dir=args.save_matrix_path)
end = time.time()
print('Evaluation time: {}s, loss: {:.5f}, accuracy: {:.5f}'.format(int(end - start), loss, accuracy))
