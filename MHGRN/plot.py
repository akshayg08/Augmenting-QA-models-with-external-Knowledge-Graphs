import os
import re
import pickle
from gen_graph import graph

train_command = "python -u grn.py -k 3 \
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
	-ebs 1 "

os.system(train_command)

eval_command = "python -u grn.py -k 3 \
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
	--mode eval \
	-ebs 1 | grep test_acc"

accuracy = []
for k in range(0, 100, 10):
	temp = 0
	print(k)
	graph(k)
	for i in range(3):
		out = os.popen(eval_command).read()
		acc = float(re.findall('\d*\.?\d+',out)[1])
		print(acc)
		temp += acc
	temp = temp/3
	accuracy.append(temp)

pickle.dump(accuracy, open("./obqa_acc.pkl", "wb"))
