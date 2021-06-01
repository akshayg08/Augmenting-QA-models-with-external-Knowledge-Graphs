# dev acc: 0.8010
python -u grn.py -k 3 \
        --unfreeze_epoch 3 \
        --format fairseq \
        --fix_trans \
        -ih 0 \
        -enc roberta-large \
        -ds obqa \
        -mbs 8 \
        -sl 80 \
        -me 2 \
        --seed 6 \
        --ent_emb transe \
        --mode train \
        -ebs 1 \
