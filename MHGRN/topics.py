import json
import torch
import numpy as np 
from sentence_transformers import SentenceTransformer

sentences = []
with open("/home/asgoindani/MHGRN/data/obqa/grounded/test.grounded.jsonl") as f:
	for line in f:
		data = json.loads(line.strip())
		sent = data["sent"].strip()
		# sent = " ".join(line.strip().split("_"))
		sentences.append(sent)

model = SentenceTransformer("stsb-roberta-large")
embeddings = model.encode(sentences)

torch.save(torch.from_numpy(embeddings), "./sentence_embedding/obqa/test.pt")
