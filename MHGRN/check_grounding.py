from utils.paths import score_paths

# stat_path = "./data/csqa/statement/temp.statement.jsonl"
# vocab_path = './data/cpnet/concept.txt'
# pattern_path = './data/cpnet/matcher_patterns.json'
# output_path = "./temp"
# extra_concepts = "./data/csqa/extra_concepts/dev_concepts.pkl"

raw_path = "./data/csqa/paths/temp1.jsonl"
transe_ent = "./data/transe/glove.transe.sgd.ent.npy"
transe_rel = "./data/transe/glove.transe.sgd.rel.npy"
vocab = './data/cpnet/concept.txt'
score_path = "./data/csqa/paths/temp1.scores.jsonl"

score_paths(raw_path, transe_ent, transe_rel, vocab, score_path, 20)
