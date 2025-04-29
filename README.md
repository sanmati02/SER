

Place unzipped dataset into dataset/

1. Change config/bi_lstm to this if features are not extracted: 
preprocess_conf:
  feature_method: 'CustomFeature'
  method_args:
    feature_type: 'mfcc'

2. Replace train_list_features.txt to train_list and do the same for test in config/bi_lstm.py

3. Run python create_data.py

4. Run python extract_features.py --configs=configs/bi_lstm.yml --save_dir=dataset/features

5. Change config/bi_lstm to this - Replace train_list to train_list_features.txt and do the same for test

6. CUDA_VISIBLE_DEVICES=0 python train.py --configs=configs/bi_lstm.yml

7. Change eval.py 
add_arg('resume_model',     str,   'models/BiLSTM_CustomFeature/best_model/',  "模型的路径")


8. python eval.py --configs=configs/bi_lstm.yml
