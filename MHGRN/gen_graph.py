from utils.graph import generate_graph, generate_adj_data_from_grounded_concepts
from utils.triples import generate_triples_from_adj
from utils.paths import generate_path_and_graph_from_adj

def graph(K):
	nproc = 20
	grounded_path = "./data/obqa/grounded/test.grounded.jsonl"
	pruned_path = "./data/obqa/paths/test.paths.pruned.jsonl"
	cpnet_vocab = "./data/cpnet/concept.txt"
	cpnet_pruned_graph = "./data/cpnet/conceptnet.en.pruned.graph"

	test_graph = "./data/obqa/graph/test.graph.jsonl"
	test_adj = "./data/obqa/graph/test.graph.adj.pk"
	triples_test = "./data/obqa/triples/test.triples.pk"
	test_adj_path = "./data/obqa/paths/test.paths.adj.jsonl"
	nxg_adj_test = "./data/obqa/graph/test.graph.adj.jsonl"
		
	# generate_graph(grounded_path, pruned_path, cpnet_vocab, cpnet_pruned_graph, test_graph)
	generate_adj_data_from_grounded_concepts(grounded_path, cpnet_pruned_graph, cpnet_vocab, test_adj, nproc, K)
	generate_triples_from_adj(test_adj, grounded_path, cpnet_vocab, triples_test)
	generate_path_and_graph_from_adj(test_adj, cpnet_pruned_graph, test_adj_path, nxg_adj_test)

if __name__ == "__main__":
	graph(0)
