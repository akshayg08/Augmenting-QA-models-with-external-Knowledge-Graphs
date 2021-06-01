import torch 
import numpy as np 
from sklearn.metrics import pairwise_distances_chunked as pwd

train = torch.load("./obqa/train.pt").type(torch.float16)
cpnet = torch.load("./concepts.pt").type(torch.float16)

distance_matrix = pwd(train, cpnet, n_jobs=20)
cnt = 1
for tp in distance_matrix:
	indices = tp.argsort(axis=1)[:, :20]
	np.save("./train_indices"+str(cnt), indices)
	cnt += 1
# indices = distance_matrix.argsort(axis=1)[:, :20]
# np.save("./obqa/train_indices", indices)
