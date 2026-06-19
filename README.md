# User Help: Dual Framework Agent

## Overview
Welcome to the official codebase for the research paper: **Anomaly Detection in Citation Networks via Deep Structural Learning (GLAD) and Spectral Analysis**. 

This repository implements a robust, dual-framework approach designed to detect structural anomalies in directed graphs, specifically tailored for scholarly citation networks. It combines the rapid, deterministic execution of Spectral Analysis (using the Power Method) with the deep representational capabilities of Graph Neural Networks (GLAD) to isolate anomalous behavior such as coercive citations, citation cartels, self-citation chains, and paper mills.

---

## 🧭 User Guidance System

To ensure a smooth experience executing the pipeline, please follow this step-by-step guidance system. This system is designed to walk you through data formatting, environment configuration, and execution of the dual frameworks.

### Step 1: Environment Setup
The architecture is optimized for high-performance execution on CUDA-enabled environments. To run the pipeline successfully, ensure your environment is configured with the following dependencies:
* **Deep Learning**: `torch` (PyTorch with CUDA support), `torch.nn`, `torch.nn.functional`
* **GPU-Accelerated Machine Learning**: `cuml` and `cupy` (for NearestNeighbors and Local Outlier Factor calculations)
* **Data Processing & Graph Operations**: `scipy`, `numpy`, `pandas`, `networkx`
* **Standard ML**: `sklearn.ensemble.IsolationForest`

### Step 2: Data Preparation & Ingestion
The pipeline dynamically loads graph metadata and ground-truth anomalies to construct a directed citation graph. Your data must adhere strictly to the following naming conventions and schema. Place these files in your root or `/content/` directory:

1. **Graph Metadata File (`papers_metadata_{dataset_name}.csv`)**:
   * Contains the nodes and edges of your network.
   * **Required Columns**: 
     * `id`: The unique identifier for the source paper.
     * `referenced_works`: A stringified list (parsable by `ast.literal_eval`) of target node IDs cited by the source paper.
2. **Ground Truth File (`{dataset_name}_anomalies.csv`)**:
   * Contains the labels for validation.
   * **Required Columns**:
     * `id`: The unique identifiers of the known anomalous papers.

*Note: The datasets array currently expects:* `'coercive_citations', 'self_citation_chains', 'citation_cartels', 'random_connectors', 'paper_mill_star'`.

### Step 3: Execution Flow & Framework Methodology

The system is split into two primary detection phases:

#### Phase A: Spectral Structural Anomaly Detection
The function `run_spectral()` provides a deterministic baseline using the Power Method.
1. It row-normalizes the sparse adjacency matrix using inverse out-degrees.
2. Iteratively calculates the dominant eigenvector (v1) until convergence (L_inf norm `< 1e-7`).
3. Computes a continuous Z-score based on the mean and standard deviation of the eigenvector.
4. Nodes exceeding the defined threshold are flagged as structural anomalies.

#### Phase B: Learning-Based Structural Anomaly Detection (Partial GLAD)
The function `train_glad_model()` trains a Graph Neural Network utilizing PyTorch sparse tensor operations.
1. **Input Features**: Extracts local topological node features including in-degree, out-degree, in/out ratio, and 2-hop neighborhood counts.
2. **Encoder Architecture (`CitationGNNEncoder`)**: A 2-layer GNN that independently aggregates messages from *cited* (forward edges), *citing* (backward edges), and *self-loops*.
3. **Training**: Uses self-supervised link prediction with a `BCEWithLogitsLoss` criterion. Positive edges are sampled from the graph, and negative edges are randomly generated.
4. **Outputs**: Generates L2-normalized node embeddings (`embeddings_matrix`) capturing complex structural semantics.

### Step 4: Anomaly Scoring & Interpreting Results
After generating embeddings, the `compute_glad_scores()` function compares each node's embedding against its local neighborhood context (calculated via `A_reconstructed.dot(embeddings) / out_degree`).

Anomalies are ranked using four distinct metrics:
* **Euclidean**: L2 distance between the node embedding and its context.
* **Cosine**: Cosine distance (1 - similarity) between the node and its context.
* **LOF**: GPU-accelerated Local Outlier Factor (`custom_lof_gpu` / `cuml`), measuring local reachability density.
* **Isolation Forest**: Tree-based outlier detection (`sklearn`), isolating anomalies based on embedding path lengths.

The system will print convergence results grouped by $K$ (e.g., top 1%, 5%, 10% of nodes):
* **Hits**: The raw count of true anomalies successfully detected.
* **Recall**: Percentage of true anomalies found out of all ground-truth anomalies.
* **Precision**: Percentage of actual anomalies within the top $K$ flagged nodes.
* **Overlap**: Jaccard similarity between the nodes flagged by the Spectral baseline and the GLAD framework.

---

## ⚙️ Advanced Configuration

If you are extending the framework for novel datasets or modifying the core architecture, review the following tuning parameters:
* **GNN Dimensionality**: Inside `train_glad_model`, you can modify `hidden_dim` and `output_dim` (currently set to 32).
* **Optimization**: The model uses the Adam optimizer (`lr=0.01`) with a `StepLR` scheduler (`step_size=50`, `gamma=0.5`). 
* **Batch Sizing**: Adjust `batch_size` (default `100,000`) based on available VRAM. Reduce this if you encounter CUDA out-of-memory exceptions during tensor concatenation.
* **LOF Neighbors**: The `custom_lof_gpu` function defaults to `n_neighbors=20`. This can be tuned based on the average degree density of your specific graph.

## 🛠️ Troubleshooting
* **Sparse Tensor Warnings**: You may see implicit disabling of sparse invariant checks in PyTorch (`UserWarning: Sparse invariant checks...`). This is expected behavior during `torch.sparse_coo_tensor` creation and does not affect mathematical correctness.
* **Memory Errors (Colab/Jupyter)**: If RAM becomes saturated when constructing `A_train` or evaluating `scipy_csr_to_torch_sparse`, ensure that unneeded objects are garbage collected before the GLAD loop initiates.
