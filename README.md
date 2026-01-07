# Geometry-Aware Computational Framework for Prioritizing Residues Involved in Protein–Protein Interfaces

  

This repository contains a structure-based computational framework for **prioritizing residues at protein–protein interaction (PPI) interfaces** using local 3D structural information.

  

The project is motivated by practical structural biology use cases where the goal is not to exhaustively annotate all interface residues, but to **rank and highlight a limited subset of residues** that are most likely to participate in a specific protein–protein interface.

  

---

  

## What this project does

  

- Operates on **single protein chains** extracted from experimentally resolved PDB protein–protein complexes

- Assigns a **continuous interface propensity score** to each residue based on local geometric context

- Ranks residues by likelihood of interface participation

- Evaluates performance using **ranking-based metrics** (PR-AUC, Precision@K, F1@K)

- Provides **3D structural visualizations** of ground truth interfaces and top-ranked predictions

  

---

  

## What this project does NOT do

  

- Does not predict full interfaces exhaustively

- Does not identify interaction partners

- Does not perform docking or binding affinity estimation

- Does not use sequence alignments, evolutionary profiles, or pretrained language models

  

---

  

## Dataset

  

- **Source:** Protein Data Bank (PDB), mmCIF format

- **Examples:** ~1,900 chain-level examples after filtering

- **Label definition:** A residue is labeled as interfacial if any heavy atom lies within ≤ 5 Å of any heavy atom of the partner chain

- **Splitting:** Train / validation / test splits are performed at the **complex level** to avoid structural leakage

  

The partner chain is used **only to define interface labels**, and is **not available to any model at inference time**.

  

---

  

## Structural Representation

  

Each protein chain is represented as a **residue-level spatial graph**:

  

- **Nodes:** amino-acid residues

- **Edges:** residues within a fixed intra-chain distance threshold

- **Node features:** amino-acid identity, coarse physicochemical categories, solvent exposure proxy, local contact density, relative sequence position

- **Edge features:** Euclidean distance between residues

  

All features are derived from **single-chain 3D structure only**.

  

---

  

## Models Implemented

  

### Baseline 0 — Surface-exposure heuristic (partner-agnostic)

  

- Ranks residues by inverse local contact density

- Uses only intra-chain geometry

- Serves as a realistic lower-bound reference

  

### Classical machine learning baselines

  

- Logistic regression

- Random forest

- Operate on residue-level handcrafted features

  

### Geometry-aware Graph Neural Network

  

- Compact residue-level GNN (3 message-passing layers)

- Distance-weighted neighbor aggregation

- Trained as a per-residue binary classifier

- Interpreted as a ranking model during evaluation

  

---

  

## Evaluation

  

Because interface residues are sparse and interface size varies across complexes, evaluation is performed in a **ranking-based manner**:

  

- **PR-AUC** — global ranking quality under class imbalance

- **Precision@K and F1@K** (K = 10, 20, 30) — enrichment under a fixed residue inspection budget

  

Top-K evaluation reflects realistic scenarios where only a limited number of residues can be examined experimentally or structurally.

  

---

  

## Visualizations and Case Studies

  

The primary qualitative output of the project is a set of **3D structural case studies**, showing:

  

- Full ground-truth interface regions for context

- Top-K predicted residues mapped onto protein structures

- Spatial distribution of true positives, false positives, and false negatives

  

These visualizations are intended to support **structural interpretation**, not just metric-based comparison.

  
## Repository Structure



protein-interface-prioritization/  
├── README.md  
├── environment.yml  
├── LICENSE  
│  
├── data/  
│ ├── raw/ # not tracked  
│ ├── processed/  
│ └── metadata/  
│ └── complex_list.csv  
│  
├── src/  
│ ├── preprocessing/  
│ ├── baselines/  
│ ├── models/  
│ ├── evaluation/  
│ └── visualization/  
│  
├── notebooks/  
│ ├── 01_preprocessing.ipynb  
│ ├── 02_baseline_heuristic.ipynb  
│ ├── 03_classical_ml.ipynb  
│ ├── 04_gnn_training.ipynb  
│ └── 05_case_studies_3d_maps.ipynb  
│  
├── results/  
│ ├── metrics/  
│ └── figures/  
│  
└── report/  
└── final_report.pdf
---

  

## Reproducibility

  

All experiments can be reproduced using the notebooks provided.

The repository avoids tracking large raw structure files; PDB structures can be re-downloaded as needed.

  

---
