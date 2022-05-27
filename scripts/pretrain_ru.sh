#!/usr/bin/env bash

FAIRSEQ_DIR=$BASE_DIR/bert-nmt

DATA_DIR=/scratch/project_2002016/datasets/data-gec
PROCESSED_DIR=$DATA_DIR/process/pseudodata
MODEL_DIR=/scratch/project_2002016/gec_model_pretrained


CUDA_VISIBLE_DEVICES=0,1,2,3 python -u $FAIRSEQ_DIR/train.py $PROCESSED_DIR/bin \
    --save-dir $MODEL_DIR \
    --arch transformer_iwslt_de_en \
    --weight-decay 0.0001 \
    --max-tokens 4096 \
    --optimizer adam \
    --lr 0.001 \
    --min-lr '1e-09' \
    -s src \
    -t trg \
    --dropout 0.3 \
    --lr-scheduler inverse_sqrt \
    --lr-shrink 0.7 \
    --min-lr 1e-06 \
    --clip-norm 0.1 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-update 7400 \
    --warmup-updates 370 \
    --warmup-init-lr '1e-07' \
    --max-epoch 30 \
    --update-freq 8 \
    --ddp-backend c10d \
    --validate-interval 1 --patience 10 --save-interval 2 --keep-interval-updates 10 \
    --adam-betas '(0.9,0.98)' \
    --log-format simple \
    --reset-optimizer \
    --reset-meters \
    --reset-dataloader \
    --seed $seed \
    --share-all-embeddings \
    --task translation \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric