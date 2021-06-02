from utils import graph

cpnet_graph = "./data/cpnet/conceptnet.en.pruned.graph"
cpnet_vocab = "./data/cpnet/concept.txt"

grnd = "./data/csqa/grounded/test.grounded.jsonl"
output = "./data/csqa/graph/test.graph.adj.pk"
extra_concepts = "./data/csqa/extra_concepts/test_concepts.pkl"

graph.generate_adj_data_from_grounded_concepts__use_LM(grnd, cpnet_graph, cpnet_vocab, output, extra_concepts, 70)

grnd = "./data/csqa/grounded/dev.grounded.jsonl"
output = "./data/csqa/graph/dev.graph.adj.pk"
extra_concepts = "./data/csqa/extra_concepts/dev_concepts.pkl"

graph.generate_adj_data_from_grounded_concepts__use_LM(grnd, cpnet_graph, cpnet_vocab, output, extra_concepts, 70)

grnd = "./data/csqa/grounded/train.grounded.jsonl"
output = "./data/csqa/graph/train.graph.adj.pk"
extra_concepts = "./data/csqa/extra_concepts/train_concepts.pkl"

graph.generate_adj_data_from_grounded_concepts__use_LM(grnd, cpnet_graph, cpnet_vocab, output, extra_concepts, 70)

# grnd = "./data/csqa/grounded/temp"
# output = "./temp"
# extra_concepts = "./data/csqa/extra_concepts/test_concepts.pkl"

# graph.generate_adj_data_from_grounded_concepts__use_LM(grnd, cpnet_graph, cpnet_vocab, output, extra_concepts, 20)

