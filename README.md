# Augmenting-QA-models-with-external-Knowledge-Graphs

## Dependencies

- [Python](<https://www.python.org/>) >= 3.6
- [PyTorch](<https://pytorch.org/get-started/locally/>) == 1.1.0
- [transformers](<https://github.com/huggingface/transformers/tree/v2.0.0>) == 2.0.0
- [tqdm](<https://github.com/tqdm/tqdm>)
- [dgl](<https://github.com/dmlc/dgl>) == 0.3.1 (GPU version)
- [networkx](<https://networkx.github.io/>) == 2.3

```bash
conda create -n krqa python=3.6 numpy matplotlib ipython
source activate krqa
conda install pytorch=1.1.0 torchvision cudatoolkit=10.0 -c pytorch
pip install dgl-cu100==0.3.1
pip install transformers==2.0.0 tqdm networkx==2.3 nltk spacy==2.1.6
python -m spacy download en
```
## Usage

### 1. Download Data

First, you need to download all the necessary data in order to train the model:

```bash
git clone https://github.com/INK-USC/MHGRN.git
cd MHGRN
bash scripts/download.sh
```

The script will:

- Download the [CommonsenseQA](<https://www.tau-nlp.org/commonsenseqa>) dataset
- Download [ConceptNet](<http://conceptnet.io/>)
- Download pretrained TransE embeddings

### 2. Preprocess

To preprocess the data, run:

```bash
python preprocess.py
```

By default, all available CPU cores will be used for multi-processing in order to speed up the process. Alternatively, you can use "-p" to specify the number of processes to use:

```bash
python preprocess.py -p 20
```

The script will:

- Convert the original datasets into .jsonl files (stored in `data/csqa/statement/`)
- Extract English relations from ConceptNet, merge the original 42 relation types into 17 types
- Identify all mentioned concepts in the questions and answers
- Extract subgraphs for each q-a pair

### 3. Extract Extra Concepts

To get the embeddings for the QA pairs and concepts run

```bash
python topics.py
```

After getting the embeddings, caclulate the nearest neighbors by running

```bash
python ./sentence_embedding/create_indices.py
```
Finally save the extra concepts of each QA pair, by running:

```bash
python ./sentence_embedding/extract_concepts.py
```

## Training

To replicate the results from the MHGRN paper, run the below command

```bash 
bash bash scripts/run_grn_csqa.sh
```

To observe the accuracy drop by removing K% nodes, run the following command. This script will remove K percent nodes and train the model and finally save accuracies for K varying from 0 to 100 with step size of 10.

```bash 
python plot.py
```

