export BERT_BASE_DIR=./bert-mini
python convert_checkpoint.py \
  --tf_checkpoint $BERT_BASE_DIR/bert_model.ckpt-100000 \
  --config $BERT_BASE_DIR/bert_config.json \
  --pytorch_dump_output $BERT_BASE_DIR/pytorch_model.bin