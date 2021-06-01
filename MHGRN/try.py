import json

with open("./data/csqa/paths/test.paths.raw.jsonl") as f:
	for line in f:
		data = json.loads(line.strip())
		print(len(data))
		for i in data:
			print(i)

		break
