import pickle
import numpy as np 

with open("/home/asgoindani/MHGRN/data/cpnet/concept.txt") as f:
	cpnet = f.readlines()

cpnet = np.array([i.strip() for i in cpnet])
concepts = []
a = np.load("./test_indices.npy")
cp = cpnet[a]

for j in cp:
	concepts.append(list(j))

print(concepts[5])
pickle.dump(concepts, open("./test_concepts.pkl", "wb"))
