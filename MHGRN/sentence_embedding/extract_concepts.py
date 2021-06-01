import pickle
import numpy as np

with open("/home/asgoindani/MHGRN/data/cpnet/concept.txt") as f:
	cpnet = f.readlines()

cpnet = np.array([i.strip() for i in cpnet])
concepts = []
# for i in range(1, 120):
# 	a = np.load("./obqa/train_indices"+str(i)+".npy")
# 	cps = cpnet[a]
# 	for j in cps:
# 		concepts.append(list(j))

# pickle.dump(concepts, open("./obqa/train_concepts.pkl", "wb"))

a = np.load("./obqa/test_indices.npy")
cps = cpnet[a]
for j in cps:
	concepts.append(list(j))

pickle.dump(concepts, open("./obqa/test_concepts.pkl", "wb"))

