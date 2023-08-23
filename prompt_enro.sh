#!/bin/bash

source activate /bigstore/hlcm2/tianzhiliang/test/software/anaconda3_4_4_0_pytorch0_1_12/envs/unmt
#source activate /home/laizhiquan/.conda/envs/fairseq
script=./train.py
log=./log/en-ro_mlm_ft_eval_xlmr_y4bleu_y4ppl_decy4_enc1
pt_model="/data/tianzhiliang/psl/LM/MASS/enro/mass_enro_1024.pth,/data/tianzhiliang/psl/LM/MASS/enro/mass_enro_1024.pth"
pt_model2="/data/tianzhiliang/psl/LM/XLM/enro/mlm_enro_1024.pth,/data/tianzhiliang/psl/LM/XLM/enro/mlm_enro_1024.pth"
ft_model="/data/tianzhiliang/psl/LM/MASS/enro/mass_ft_enro_1024.pth,/data/tianzhiliang/psl/LM/MASS/enro/mass_ft_enro_1024.pth"
xlmr_model="/data/tianzhiliang/psl/LM/XLMR"
codes_path="/data/tianzhiliang/psl/unmt/MASS/data/codes-vocab/codes_enro"
vocab_path="/data/tianzhiliang/psl/unmt/MASS/data/codes-vocab/vocab_enro"
checkpoint="/dat01/laizhiquan/psl/Project/UNMT/MASS/dumped/unsupMT_enro/prompt_finetune.2_y4bleu_y4ppl/checkpoint.pth,/dat01/laizhiquan/psl/Project/UNMT/MASS/dumped/unsupMT_enro/prompt_finetune.2_y4bleu_y4ppl/checkpoint.pth"

python $script \
--eval_only true \
--cuda true \
--prefix true \
--exp_name unsupMT_enro \
--exp_id mlm_ft_eval_xlmr_y4bleu_y4ppl_decy4_enc1 \
--data_path ./data/processed/en-ro/ \
--lgs "en-ro" \
--bt_steps "ro-en-ro,en-ro-en" \
--encoder_only false \
--emb_dim 1024 \
--n_layers 6 \
--n_heads 8 \
--dropout 0.1 \
--attention_dropout 0.1 \
--gelu_activation true \
--tokens_per_batch 2000 \
--batch_size 128 \
--bptt 256 \
--optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
--epoch_size 100000 \
--max_epoch 1000 \
--eval_bleu true \
--reload_model \
$ft_model \
--validation_metrics \
test_ro-en_mt_bleu_y3,test_ro-en_mt_bleu_y4 \
--stopping_criterion \
test_ro-en_mt_bleu_y4,10 \
--multiLM $xlmr_model \
--codes_path $codes_path \
--vocab_path $vocab_path
