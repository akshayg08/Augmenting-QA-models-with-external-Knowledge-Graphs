import torch
import numpy as np 

def distance(filename):
	if "dev" in filename or "test" in filename:
		from sklearn.metrics import pairwise_distances as pwd
		data = torch.load(filename)
		distance_matrix = pwd(data.cpu().numpy(), cpnet, n_jobs=20)
		indices = distance_matrix.argsort(axis=1)[:, :20]
		np.save(filename.replace(".pt", "_indices"), indices)

	else:
		from sklearn.metrics import pairwise_distances_chunked as pwd
		train = torch.load(filename)
		distance_matrix = pwd(train.cpu().numpy(), cpnet, n_jobs=20)
		cnt = 1
		for tp in distance_matrix:
			indices = tp.argsort(axis=1)[:, :20]
			np.save(filename.replace(".pt", "_indices")+str(cnt), indices)
			cnt += 1

cpnet = np.load("./data/cpnet/tzw.ent.npy")
distance("./data/csqa/test_emb.pt")
