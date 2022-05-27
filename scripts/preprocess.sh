#!/usr/bin/env bash


FAIRSEQ_DIR=$BASE_DIR/bert-nmt

DATA_DIR=/scratch/project_2002016/datasets/data-gec
VOCAB_DIR=$DATA_DIR/vocab

PROCESSED_DIR=$DATA_DIR/process/pseudodata


cpu_num=`grep -c ^processor /proc/cpuinfo`


python $FAIRSEQ_DIR/preprocess.py --source-lang src --target-lang trg \
    --trainpref $PROCESSED_DIR/train \
    --validpref $PROCESSED_DIR/valid \
    --testpref $PROCESSED_DIR/test \
    --destdir $PROCESSED_DIR/bin \
    --srcdict $VOCAB_DIR/dict.src \
    --tgtdict $VOCAB_DIR/dict.trg \
    --workers $cpu_num \





