#!/usr/bin/env bash


#FAIRSEQ_DIR=$BASE_DIR/bert-nmt
#FAIRSEQ_DIR=/projappl/project_2002016/fairseq/fairseq_cli
BASE_DIR=/projappl/project_2002016
FAIRSEQ_DIR=$BASE_DIR/bert-nmt
DATA_DIR=/scratch/project_2002016/datasets/data-gec
bert_model=$BASE_DIR/gramcor/bert-pretraned/rubert_cased_L-12_H-768_A-12_pt


PROCESSED_DIR=$DATA_DIR/process/pseudodata


cpu_num=`grep -c ^processor /proc/cpuinfo`


train_src=$PROCESSED_DIR/train.src
train_trg=$PROCESSED_DIR/train.trg
valid_src=$PROCESSED_DIR/valid.src
valid_trg=$PROCESSED_DIR/valid.trg
test_src=$PROCESSED_DIR/test.src
test_trg=$PROCESSED_DIR/test.trg

cd PROCESSED_DIR
cp $train_src $PROCESSED_DIR/train.bert.src
cp $valid_src $PROCESSED_DIR/valid.bert.src
cp $test_src $PROCESSED_DIR/test.bert.src
cp $train_trg $PROCESSED_DIR/train.bert.trg
cp $valid_trg $PROCESSED_DIR/valid.bert.trg
cp $test_trg $PROCESSED_DIR/test.bert.trg

python $FAIRSEQ_DIR/preprocess.py --source-lang src --target-lang trg \
    --trainpref $PROCESSED_DIR/train \
    --validpref $PROCESSED_DIR/valid \
    --testpref $PROCESSED_DIR/test \
    --destdir $PROCESSED_DIR/bin \
    --srcdict $PROCESSED_DIR/dict.src \
    --tgtdict $PROCESSED_DIR/dict.trg \
    --workers $cpu_num \
    --bert-model-name $bert_model






