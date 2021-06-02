import json
import torch
from transformers import BertModel, BertTokenizer

device="cuda"

model = BertModel.from_pretrained("bert-large-uncased").to(device)
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
model.eval()

sentences = []
with open("./data/csqa/grounded/train.grounded.jsonl") as f:
	for line in f:
		data = json.loads(line.strip())
		sentences.append(data["sent"].lower().strip())

batch_size = 512
vecs = []
cnt = 0
for i in range(0, len(sentences), batch_size):
	cnt += 1
	print(cnt)
	batch = sentences[i : i+batch_size]
	# print(len(batch))
	tokens = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
	with torch.no_grad():
		outputs = model(**tokens)
		vecs.append(outputs.last_hidden_state[:,0,:].to(device))

vecs = torch.cat(vecs, dim=0).to(device)
print(vecs.shape)
torch.save(vecs, "./train_emb.pt")
