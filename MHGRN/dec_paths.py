import json

K = 0

f1 = open("./data/csqa/paths/small_ih_test.paths.pruned.jsonl", "w")

with open("./data/csqa/paths/ih_test.paths.pruned.jsonl") as f:
	for line in f:
		data = json.loads(line.strip())
		for pair in data:
			if pair["pf_res"] is None:
				pair["pf_res"] = []
				continue
				
			pair["pf_res"] = pair["pf_res"][:K]

		temp = json.dumps(data)
		f1.write(temp + "\n")

f1.close()

