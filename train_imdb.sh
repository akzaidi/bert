#!/usr/bin/env bash

BERT_BASE_DIR='/home/alizaidi/nlpdev/BERT_implementations/pretrained-models/uncased_L-12_H-768_A-12'
DATASETS='/home/alizaidi/data/glue_data/'
OUTPUTS='/home/alizaidi/nlpdev/BERT_implementations/bert-imdb/outputs'

tensorboard --logdir $OUTPUTS &

python run_classifier.py \
  --task_name=IMDB \
  --do_train=true \
  --do_eval=true \
  --data_dir=$DATASETS/IMDB \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=256 \
  --train_batch_size=16 \
  --learning_rate=2e-5 \
  --num_train_epochs=3 \
  --output_dir=$OUTPUTS \
  $@