"""Compare DeepWalk and a two-layer GCN on local Cora and Citeseer files.

Put the extracted files in data/cora/ and data/citeseer/ (see README.md).
Run from Command Prompt: py run_project.py --epochs 1000 --seed 42 --num-seeds 5
Default outputs: results.csv and results_summary.csv.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split


@dataclass
class GraphData:
    name: str
    features: np.ndarray
    labels: np.ndarray
    adjacency: sparse.csr_matrix
    class_names: np.ndarray
    ignored_citations: int


def load_graph(data_dir: Path, name: str) -> GraphData:
    """Read paper features/labels and citations from the original-format files."""
    folder = data_dir / name
    content_file = folder / f"{name}.content"
    cites_file = folder / f"{name}.cites"
    for path in (content_file, cites_file):
        if not path.is_file():
            raise FileNotFoundError(f"Cannot find {path}. See README.md for folder setup.")

    # First column: paper ID; last column: topic; middle columns: word indicators.
    papers = pd.read_csv(content_file, sep="\t", header=None, dtype={0: str})
    if papers.shape[1] < 3:
        raise ValueError(f"{content_file} should contain IDs, word features, and labels")
    paper_ids = papers.iloc[:, 0].astype(str).tolist()
    if len(paper_ids) != len(set(paper_ids)):
        raise ValueError(f"Duplicate paper IDs found in {content_file}")
    id_to_index = {paper_id: i for i, paper_id in enumerate(paper_ids)}
    features = papers.iloc[:, 1:-1].to_numpy(dtype=np.float32)
    # Normalize each paper's word vector; zero rows remain zero.
    features /= np.maximum(features.sum(axis=1, keepdims=True), 1.0)
    class_names, labels = np.unique(papers.iloc[:, -1].astype(str), return_inverse=True)
    labels = labels.astype(np.int64)

    citations = pd.read_csv(cites_file, sep="\t", header=None, dtype=str)
    if citations.shape[1] != 2:
        raise ValueError(f"{cites_file} should contain two paper IDs per line")

    # Treat a citation as an undirected connection. Some CiteSeer citations
    # mention papers that have no feature/label row; exclude those connections.
    edges = set()
    ignored = 0
    for first, second in citations.itertuples(index=False, name=None):
        u, v = id_to_index.get(first), id_to_index.get(second)
        if u is None or v is None:
            ignored += 1
            continue
        if u != v:
            edges.add((min(u, v), max(u, v)))

    n = len(paper_ids)
    rows = [u for u, v in edges] + [v for u, v in edges]
    cols = [v for u, v in edges] + [u for u, v in edges]
    adjacency = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n)
    )
    adjacency.sort_indices()
    return GraphData(name, features, labels, adjacency, class_names, ignored)


def split_nodes(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use exactly the same stratified 60/20/20 split for both methods."""
    indices = np.arange(len(labels))
    train, remainder = train_test_split(
        indices, test_size=0.4, stratify=labels, random_state=seed
    )
    validation, test = train_test_split(
        remainder, test_size=0.5, stratify=labels[remainder], random_state=seed
    )
    return train, validation, test


def evaluate(labels: np.ndarray, test: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    """Multiclass accuracy and macro one-vs-rest ROC AUC on unseen labels."""
    truth = labels[test]
    accuracy = accuracy_score(truth, probabilities.argmax(axis=1))
    auc = roc_auc_score(
        truth,
        probabilities,
        labels=np.arange(probabilities.shape[1]),
        multi_class="ovr",
        average="macro",
    )
    return float(accuracy), float(auc)


def train_deepwalk(
    graph: GraphData, train: np.ndarray, validation: np.ndarray, test: np.ndarray,
    seed: int,
) -> tuple[float, float, int, float, float]:
    """Tune skip-gram epochs and classifier C on validation log loss."""
    from gensim.models import Word2Vec

    n = len(graph.labels)
    tokens = [str(node) for node in range(n)]
    rng = np.random.default_rng(seed)
    walks: list[list[str]] = []
    for _ in range(10):  # ten random walks starting at each paper
        for start in rng.permutation(n):
            current = int(start)
            walk = [tokens[current]]
            for _ in range(39):  # at most forty papers in each walk
                nearby = graph.adjacency.indices[
                    graph.adjacency.indptr[current] : graph.adjacency.indptr[current + 1]
                ]
                if not len(nearby):
                    break
                current = int(nearby[rng.integers(len(nearby))])
                walk.append(tokens[current])
            walks.append(walk)

    # Reuse the same walks and split for every candidate. Only the training
    # labels fit the classifier; validation labels select the hyperparameters.
    best_loss = float("inf")
    best_embeddings = None
    best_classifier = None
    best_epochs, best_c = 0, 0.0
    for word2vec_epochs in (5, 10):
        # Skip-gram with hierarchical softmax is the representation-learning step.
        word_model = Word2Vec(
            sentences=walks,
            vector_size=64,
            window=5,
            min_count=1,
            sg=1,
            hs=1,
            negative=0,
            sample=0,
            workers=1,
            epochs=word2vec_epochs,
            seed=seed,
        )
        embeddings = np.stack([word_model.wv[token] for token in tokens])
        for c in (0.1, 1.0, 10.0):
            classifier = LogisticRegression(C=c, max_iter=1000, random_state=seed)
            classifier.fit(embeddings[train], graph.labels[train])
            validation_probabilities = classifier.predict_proba(embeddings[validation])
            validation_loss = log_loss(
                graph.labels[validation], validation_probabilities,
                labels=classifier.classes_,
            )
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_embeddings = embeddings
                best_classifier = classifier
                best_epochs, best_c = word2vec_epochs, c

    # Evaluate the selected candidate on the test set exactly once.
    probabilities = best_classifier.predict_proba(best_embeddings[test])
    accuracy, auc = evaluate(graph.labels, test, probabilities)
    return accuracy, auc, best_epochs, best_c, best_loss


def normalized_adjacency(adjacency: sparse.csr_matrix) -> sparse.csr_matrix:
    """D^(-1/2) (A + I) D^(-1/2), as used in a GCN."""
    with_loops = adjacency + sparse.eye(adjacency.shape[0], dtype=np.float32, format="csr")
    degrees = np.asarray(with_loops.sum(axis=1)).reshape(-1)
    inverse_sqrt = np.power(degrees, -0.5).astype(np.float32)
    degree_matrix = sparse.diags(inverse_sqrt, format="csr")
    return (degree_matrix @ with_loops @ degree_matrix).tocsr()


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    probabilities = softmax(logits)
    selected = probabilities[np.arange(len(labels)), labels]
    return float(-np.log(np.maximum(selected, 1e-12)).mean())


def train_gcn(
    graph: GraphData, train: np.ndarray, validation: np.ndarray, test: np.ndarray,
    seed: int, max_epochs: int = 1000,
) -> tuple[float, float, int, int]:
    """Train a two-layer GCN using full-batch gradients and Adam in NumPy."""
    rng = np.random.default_rng(seed)
    adjacency = normalized_adjacency(graph.adjacency)
    # X and A are known for all papers; labels outside train are hidden from training.
    # The word features are mostly zero; keep the first aggregation sparse so
    # repeated training across several seeds and up to 1,000 epochs is practical.
    neighbor_features = adjacency @ sparse.csr_matrix(graph.features)
    _, input_width = graph.features.shape
    hidden_width, output_width = 32, len(graph.class_names)
    weights_0 = rng.normal(0, np.sqrt(2 / input_width), (input_width, hidden_width)).astype(np.float32)
    bias_0 = np.zeros(hidden_width, dtype=np.float32)
    weights_1 = rng.normal(0, np.sqrt(2 / hidden_width), (hidden_width, output_width)).astype(np.float32)
    bias_1 = np.zeros(output_width, dtype=np.float32)
    parameters = [weights_0, bias_0, weights_1, bias_1]
    moments = [np.zeros_like(parameter) for parameter in parameters]
    squares = [np.zeros_like(parameter) for parameter in parameters]
    learning_rate, weight_decay, dropout_rate = 0.01, 5e-4, 0.5
    best_validation_loss, best_epoch, stale_epochs = float("inf"), 0, 0
    best_parameters = [parameter.copy() for parameter in parameters]

    for epoch in range(1, max_epochs + 1):
        # Forward pass: softmax(A ReLU(A X W0 + b0) W1 + b1).
        hidden_pre = neighbor_features @ weights_0 + bias_0
        hidden = np.maximum(hidden_pre, 0)
        mask = (rng.random(hidden.shape) >= dropout_rate).astype(np.float32) / (1 - dropout_rate)
        hidden_neighbors = adjacency @ (hidden * mask)
        logits = hidden_neighbors @ weights_1 + bias_1

        # Cross-entropy derivatives use only the training node labels.
        delta = np.zeros_like(logits)
        delta[train] = softmax(logits[train])
        delta[train, graph.labels[train]] -= 1
        delta[train] /= len(train)
        grad_w1 = hidden_neighbors.T @ delta + weight_decay * weights_1
        grad_b1 = delta.sum(axis=0)
        grad_hidden = (adjacency.T @ (delta @ weights_1.T)) * mask
        grad_hidden *= hidden_pre > 0
        grad_w0 = neighbor_features.T @ grad_hidden + weight_decay * weights_0
        grad_b0 = grad_hidden.sum(axis=0)

        # Adam optimizer: adapt each parameter's step using past gradients.
        for parameter, gradient, moment, square in zip(
            parameters, (grad_w0, grad_b0, grad_w1, grad_b1), moments, squares
        ):
            moment *= 0.9
            moment += 0.1 * gradient
            square *= 0.999
            square += 0.001 * gradient * gradient
            corrected_moment = moment / (1 - 0.9**epoch)
            corrected_square = square / (1 - 0.999**epoch)
            parameter -= learning_rate * corrected_moment / (np.sqrt(corrected_square) + 1e-8)

        # Evaluate without dropout; use validation labels only for early stopping.
        hidden_eval = np.maximum(neighbor_features @ weights_0 + bias_0, 0)
        validation_logits = (adjacency @ hidden_eval) @ weights_1 + bias_1
        regularization = 0.5 * weight_decay * (np.sum(weights_0**2) + np.sum(weights_1**2))
        validation_loss = cross_entropy(
            validation_logits[validation], graph.labels[validation]
        ) + regularization
        if validation_loss < best_validation_loss - 1e-5:
            best_validation_loss, best_epoch, stale_epochs = validation_loss, epoch, 0
            best_parameters = [parameter.copy() for parameter in parameters]
        else:
            stale_epochs += 1
            if stale_epochs >= 30:
                break

    weights_0, bias_0, weights_1, bias_1 = best_parameters
    best_hidden = np.maximum(neighbor_features @ weights_0 + bias_0, 0)
    test_logits = ((adjacency @ best_hidden) @ weights_1 + bias_1)[test]
    accuracy, auc = evaluate(graph.labels, test, softmax(test_logits))
    return accuracy, auc, best_epoch, epoch


def summarize_results(results: list[dict]) -> list[dict]:
    """Summarize the repeated runs separately for each dataset and method."""
    summary = []
    for dataset, method in dict.fromkeys((row["dataset"], row["method"]) for row in results):
        runs = [row for row in results if row["dataset"] == dataset and row["method"] == method]
        accuracies = np.array([row["test_accuracy"] for row in runs], dtype=float)
        aucs = np.array([row["test_auc_macro_ovr"] for row in runs], dtype=float)
        ddof = 1 if len(runs) > 1 else 0
        summary.append({
            "dataset": dataset,
            "method": method,
            "num_seeds": len(runs),
            "accuracy_mean": f"{accuracies.mean():.6f}",
            "accuracy_std": f"{accuracies.std(ddof=ddof):.6f}",
            "auc_mean": f"{aucs.mean():.6f}",
            "auc_std": f"{aucs.std(ddof=ddof):.6f}",
        })
    return summary


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=script_dir / "data")
    parser.add_argument("--output", type=Path, default=script_dir / "results.csv")
    parser.add_argument("--summary-output", type=Path, default=script_dir / "results_summary.csv")
    parser.add_argument("--dataset", choices=["cora", "citeseer", "both"], default="both")
    parser.add_argument("--method", choices=["deepwalk", "gcn", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=1000, help="GCN epoch limit")
    parser.add_argument("--seed", type=int, default=42, help="First random seed")
    parser.add_argument("--num-seeds", type=int, default=5, help="Number of consecutive seeds")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.num_seeds < 1:
        parser.error("--num-seeds must be at least 1")
    if args.seed < 0:
        parser.error("--seed must be at least 0")
    if args.output.resolve() == args.summary_output.resolve():
        parser.error("--output and --summary-output must be different files")

    results = []
    names = ("cora", "citeseer") if args.dataset == "both" else (args.dataset,)
    for name in names:
        graph = load_graph(args.data_dir, name)
        print(f"\n{name}: {len(graph.labels)} papers, {graph.adjacency.nnz // 2} links, "
              f"{len(graph.class_names)} topics; {graph.ignored_citations} unmatched citations ignored")
        methods = ("deepwalk", "gcn") if args.method == "both" else (args.method,)
        for seed in range(args.seed, args.seed + args.num_seeds):
            train, validation, test = split_nodes(graph.labels, seed)
            print(f"Seed {seed}: {len(train)} train / {len(validation)} validation / "
                  f"{len(test)} test")
            for method in methods:
                print(f"Training {method} on {name} (seed {seed})...", flush=True)
                if method == "deepwalk":
                    accuracy, auc, walk_epochs, classifier_c, walk_val_loss = train_deepwalk(
                        graph, train, validation, test, seed
                    )
                    best_epoch, trained_epochs = "", ""
                else:
                    accuracy, auc, best_epoch, trained_epochs = train_gcn(
                        graph, train, validation, test, seed=seed, max_epochs=args.epochs
                    )
                    walk_epochs, classifier_c, walk_val_loss = "", "", ""
                print(f"  Test accuracy: {accuracy:.4f}; test AUC: {auc:.4f}" +
                      (f"; best GCN epoch: {best_epoch}; trained epochs: {trained_epochs}"
                       if method == "gcn" else
                       f"; Word2Vec epochs: {walk_epochs}; C: {classifier_c:g}; "
                       f"validation loss: {walk_val_loss:.4f}"))
                results.append({
                    "dataset": name,
                    "method": method,
                    "seed": seed,
                    "test_accuracy": accuracy,
                    "test_auc_macro_ovr": auc,
                    "train_nodes": len(train),
                    "validation_nodes": len(validation),
                    "test_nodes": len(test),
                    "best_gcn_epoch": best_epoch,
                    "trained_gcn_epochs": trained_epochs,
                    "deepwalk_word2vec_epochs": walk_epochs,
                    "deepwalk_classifier_c": classifier_c,
                    "deepwalk_validation_loss": walk_val_loss,
                })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved scores to {args.output}")

    summary = summarize_results(results)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_output.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Saved averages to {args.summary_output}")
    for row in summary:
        print(f"{row['dataset']} {row['method']} ({row['num_seeds']} seeds): "
              f"accuracy {float(row['accuracy_mean']):.4f} ± {float(row['accuracy_std']):.4f}; "
              f"AUC {float(row['auc_mean']):.4f} ± {float(row['auc_std']):.4f}")


if __name__ == "__main__":
    main()
