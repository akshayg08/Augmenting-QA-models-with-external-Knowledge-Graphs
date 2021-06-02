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
bash scripts/run_grn_csqa.sh
```

To observe the accuracy drop by removing K% nodes, run the following command. This script will remove K percent nodes and train the model and finally save accuracies for K varying from 0 to 100 with step size of 10.

```bash 
python plot.py
```

## QAGNN Model

### 0. Dependencies

- [Python](<https://www.python.org/>) == 3.7
- [PyTorch](<https://pytorch.org/get-started/locally/>) == 1.4.0
- [transformers](<https://github.com/huggingface/transformers/tree/v2.0.0>) == 2.0.0
- [torch-geometric](https://pytorch-geometric.readthedocs.io/) ==1.6.0

Run the following commands to create a conda environment (assuming CUDA10.1):
```bash
conda create -n qagnn python=3.7
source activate qagnn
pip install numpy==1.18.3 tqdm
pip install torch==1.4.0 torchvision==0.5.0
pip install transformers==2.0.0 nltk spacy==2.1.6
python -m spacy download en

#for torch-geometric
pip install torch-scatter==2.0.4 -f https://pytorch-geometric.com/whl/torch-1.4.0+cu101.html
pip install torch-cluster==1.5.4 -f https://pytorch-geometric.com/whl/torch-1.4.0+cu101.html
pip install torch-sparse==0.6.1 -f https://pytorch-geometric.com/whl/torch-1.4.0+cu101.html
pip install torch-spline-conv==1.2.0 -f https://pytorch-geometric.com/whl/torch-1.4.0+cu101.html
pip install torch-geometric==1.6.0 -f https://pytorch-geometric.com/whl/torch-1.4.0+cu101.html
```


### 1. Download Data

Download all the raw data -- ConceptNet, CommonsenseQA, OpenBookQA -- by
```
./qagnn/download_raw_data.sh
```

You can preprocess the raw data by running
```
python ./qagnn/preprocess.py -p <num_processes>
```
The script will:
* Setup ConceptNet (e.g., extract English relations from ConceptNet, merge the original 42 relation types into 17 types)
* Convert the QA datasets into .jsonl files (e.g., stored in `data/csqa/statement/`)
* Identify all mentioned concepts in the questions and answers
* Extract subgraphs for each q-a pair

**TL;DR**. The preprocessing may take long; for your convenience, you can download all the processed data by
```
./qagnn/download_preprocessed_data.sh
```
### 2. Training
For CommonsenseQA, run
```
./run_qagnn__csqa.sh
