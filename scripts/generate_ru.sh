#!/usr/bin/env bash

bert_type=bert-base-russian-cased
seed=2222
BASE_DIR=/projappl/project_2002016

SUBWORD_NMT=$BASE_DIR/subword-nmt
FAIRSEQ_DIR=$BASE_DIR/bert-nmt


BPE_MODEL_DIR=$BASE_DIR/bpe
DATA_DIR=/scratch/project_2002016/datasets/data-gec
VOCAB_DIR=$DATA_DIR/vocab

PROCESSED_DIR=$DATA_DIR/process
MODEL_DIR=$BASE_DIR/gec_model


$SUBWORD_NMT/apply_bpe.py -c $BPE_MODEL_DIR/codes.txt < $input > $MODEL_DIR/test.bpe.src

python -u detok.py $input $MODEL_DIR/test.bert.src
paste -d "\n" $MODEL_DIR/test.bpe.src $MODEL_DIR/test.bert.src > $MODEL_DIR/test.cat.src

echo Generating...
CUDA_VISIBLE_DEVICES=$gpu python -u ${FAIRSEQ_DIR}/interactive.py $PREPROCESS \
    --path ${MODEL_DIR}/checkpoint_best.pt \
    --beam ${beam} \
    --nbest ${beam} \
    --no-progress-bar \
    -s src \
    -t trg \
    --buffer-size 1024 \
    --batch-size 32 \
    --log-format simple \
    --remove-bpe \
    --bert-model-name $bert_type \
    < $MODEL_DIR/test.cat.src > $MODEL_DIR/test.nbest.tok

cat $MODEL_DIR/test.nbest.tok | grep "^H"  | python -c "import sys; x = sys.stdin.readlines(); x = ' '.join([ x[i] for i in range(len(x)) if (i % ${beam} == 0) ]); print(x)" | cut -f3 > $MODEL_DIR/test.best.tok
sed -i '$d' $MODEL_DIR/test.best.tok
