# Cora and Citeseer node classification

This project compares **DeepWalk** and a **two-layer graph convolutional network (GCN)** for predicting the topic of a paper. It reads files extracted from the Cora and Citeseer archives on **your own computer**. The code does not require Keras or TensorFlow.

## 1. Arrange your extracted files

Download [Cora](https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz) and [Citeseer](https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz) from the LINQS datasets. Extract both `.tgz` archives. Make a `data` folder beside `run_project.py`, with a `cora` and a `citeseer` folder inside. Copy these four files from the extracted archives:

| Location beside `run_project.py` | File |
| --- | --- |
| `data/cora/` | `cora.content` and `cora.cites` |
| `data/citeseer/` | `citeseer.content` and `citeseer.cites` |

The `.content` files give paper IDs, word features, and topic labels. The `.cites` files list pairs of citing papers. You do not need to copy the `.tgz` archives into `data`.

## 2. Install packages on Windows

Open the folder containing `run_project.py` in File Explorer. Click the folder's address bar, type `cmd`, then press Enter. Run:

```bat
py -m pip install -r requirements.txt
```

`requirements.txt` lists NumPy, pandas, SciPy, scikit-learn, and Gensim. `gensim` trains the skip-gram embeddings used by DeepWalk. The GCN is implemented in NumPy and SciPy.

## 3. Run your project

In that same Command Prompt window, run:

```bat
py run_project.py --epochs 1000 --seed 42 --num-seeds 5
```

The script runs seeds **42, 43, 44, 45, and 46** on each dataset. Both methods use the same train/validation/test split for each seed. It writes `results.csv` with **20 rows** (two datasets × two methods × five seeds) and `results_summary.csv` with **four rows**. The per-run file also records the DeepWalk settings chosen on validation data. The summary contains the mean accuracy and AUC, along with their sample standard deviations; use these values for a **mean ± standard deviation** results table in your report. Copy the GitHub repository URL into your report after uploading the code. If a dataset file cannot be found, check its location and spelling.

To repeat only seed 42, run `py run_project.py --epochs 1000 --seed 42 --num-seeds 1`. You can use `--method gcn` or `--dataset cora` to run a smaller subset, or `--seed 100 --num-seeds 10` for seeds 100–109. These options overwrite the chosen output files with the new subset. To save under different names, add `--output my_results.csv --summary-output my_summary.csv` to the command.

The code can also be opened in IDLE and run with F5 using the defaults, but Command Prompt lets you select seeds and displays training progress more reliably. If your extracted datasets are elsewhere, use `py run_project.py --data-dir "C:\path\to\folder\containing\cora\and\citeseer" --epochs 1000`.

## What the program does

1. Reads local files; excludes citations whose paper ID is absent from the `.content` file.
2. For each seed, creates a class-stratified 60% training, 20% validation, and 20% test split, shared by both methods. The test labels are held back until final evaluation.
3. **DeepWalk:** walks along citation links to generate sequences of papers, trains skip-gram embeddings using Gensim Word2Vec (hierarchical softmax), then learns a logistic-regression topic classifier from training labels. For each dataset and seed, it tries Word2Vec epochs **5 and 10** and classifier `C` values **0.1, 1, and 10** (six combinations). It chooses the combination with the lowest **validation log loss**. The same walks and split are used for all six combinations; test labels are used only after selection.
4. **GCN:** uses normalized citation links and paper word features in two graph convolution layers. It trains using training labels and chooses its final checkpoint using validation loss.
5. Reports test accuracy and **macro one-versus-rest ROC AUC** from class probabilities. `deepwalk_word2vec_epochs`, `deepwalk_classifier_c`, and `deepwalk_validation_loss` record the selected DeepWalk settings and validation score for each run. `best_gcn_epoch` records the checkpoint selected by validation loss; `trained_gcn_epochs` shows how far GCN training ran. GCN stops if validation loss fails to improve for 30 epochs or reaches the epoch limit. The paper features and citation network are known for all nodes during training, a standard *transductive* node classification setup.

Main settings: first seed 42, five consecutive seeds; 64-dimensional DeepWalk embeddings, ten walks of up to 40 nodes per paper, skip-gram window 5, with epochs and classifier `C` selected on validation data; GCN hidden width 32, dropout 0.5, Adam learning rate 0.01, L2 coefficient 0.0005, up to 1,000 epochs with validation early stopping. A best epoch equal to the epoch limit means increasing the limit may matter; it does not by itself establish that the model was undertrained. The DeepWalk settings other than epochs and `C` are fixed.

The Keras Cora tutorial is a useful reference for the dataset layout, but its sample code does not itself cover both datasets, DeepWalk, and AUC: https://keras.io/examples/graph/gnn_citations/

## Method references

- Perozzi, Al-Rfou, and Skiena, *DeepWalk: Online Learning of Social Representations* (2014): https://arxiv.org/abs/1403.6652
- Kipf and Welling, *Semi-Supervised Classification with Graph Convolutional Networks* (2017): https://arxiv.org/abs/1609.02907
