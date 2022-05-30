#!/usr/bin/env bash


#FAIRSEQ_DIR=$BASE_DIR/bert-nmt
FAIRSEQ_DIR=/projappl/project_2002016/fairseq/fairseq_cli
DATA_DIR=/scratch/project_2002016/datasets/data-gec


PROCESSED_DIR=$DATA_DIR/process/pseudodata


cpu_num=`grep -c ^processor /proc/cpuinfo`


python $FAIRSEQ_DIR/preprocess.py --source-lang src --target-lang trg \
    --trainpref $PROCESSED_DIR/train \
    --validpref $PROCESSED_DIR/valid \
    --testpref $PROCESSED_DIR/test \
    --destdir $PROCESSED_DIR/bin \
    --srcdict $PROCESSED_DIR/dict.src \
    --tgtdict $PROCESSED_DIR/dict.trg \
    --workers $cpu_num \





