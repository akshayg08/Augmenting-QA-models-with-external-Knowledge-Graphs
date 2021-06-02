import torch
import pickle

concept2id = None
id2concept = None

def load_resources(cpnet_vocab_path):
	global concept2id, id2concept

	with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
		id2concept = [w.strip() for w in fin]
	concept2id = {w: i for i, w in enumerate(id2concept)}

load_resources("./data/cpnet/concept.txt")

ec = pickle.load(open("./data/csqa/extra_concepts/dev_concepts.pkl", "rb"))
for i in range(len(ec)):
	for j in range(len(ec[i])):
		ec[i][j] = concept2id[ec[i][j].strip()]

ids = torch.load("./ids.pt")

mapping = []
for i in range(len(ec)):
	temp = []
	for j in ec[i]:
		temp.append(torch.where(ids[i]-1 == j)[0].item())
	mapping.append(temp)

outputs = torch.load("./outputs.pt")
for i in range(len(ids)):
	A = outputs[i,:,:]
	sim = torch.softmax(torch.matmul(A, A.T), dim=1)
	mask = 1 - torch.eye(200).to("cuda:1")
	sim = sim * mask
	indices = sim.argsort(dim=1,descending=True)[:, :5]
	filtered = indices[mapping[i], :]
	print(ids[i][filtered])
	print(ids[i][mapping[i]])
	print(ec[i])
	break
