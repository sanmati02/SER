import argparse
import functools

from mser.predict import MSERPredictor
from mser.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'configs/bi_lstm.yml',   'Configuration file')
add_arg('use_ms_model',     str,    None,                    'Use the public Emotion2vec model from ModelScope')
add_arg('use_gpu',          bool,   True,                    'Whether to use GPU for prediction')
add_arg('audio_path',       str,    'dataset/test.wav',      'Path to the audio file')
add_arg('model_path',       str,    'models/BiLSTM_Emotion2Vec/best_model/',     'Path to the exported prediction model')
args = parser.parse_args()
print_arguments(args=args)

# Get the predictor
predictor = MSERPredictor(configs=args.configs,
                          use_ms_model=args.use_ms_model,
                          model_path=args.model_path,
                          use_gpu=args.use_gpu)

label, score = predictor.predict(audio_data=args.audio_path)

print(f'Prediction result for audio {args.audio_path}: Label = {label}, Score = {score}')
