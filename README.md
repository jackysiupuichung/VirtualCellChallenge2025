# VirtualCellChallenge2025

Predicting gene expression responses in human embryonic stem cells using AI-driven modeling of single-cell perturbation data.

### 📌 Introduction

The **Virtual Cell Challenge 2025**, hosted by the Arc Institute and sponsored by NVIDIA, 10X Genomics, and Ultima Biosciences, tasks participants with building models that predict gene expression responses in a held-out cell type—**H1 human embryonic stem cells**—using CRISPRi perturbation data from other cell types.

This project presents our solution to the challenge, combining advanced machine learning techniques with domain knowledge in single-cell biology to generalize cellular responses across contexts.

### 🧬 Methodology

Our approach consists of the following core components:

* **Model Type**: \[e.g., Transformer-based regressor / Graph Neural Network / TabNet]
* **Input Features**:

  * Pseudo-bulk expression vectors per perturbation
  * Gene ontology embeddings
  * Cell-type condition vectors
* **Training Strategy**:

  * Supervised learning on 150 training perturbations
  * Domain adaptation for H1 generalization
  * Combined loss: differential expression + global MAE + perturbation similarity
* **Regularization**: Dropout, layer normalization, and perturbation contrastive loss

Our goal is to not only achieve high accuracy but also ensure **biological interpretability** and **robustness** on unseen contexts.


### 📂 Data

We use the official challenge datasets:

* ✅ **Training Set**: 150 CRISPRi perturbations across diverse cell types
* 📊 **Validation Set**: 50 perturbations for live leaderboard evaluation
* 🔒 **Final Test Set**: 100 perturbations on H1 cells (released Oct 2025)

We also incorporated external perturb-seq datasets such as:

* Tahoe-100M
* scBaseCount
* Replogle et al., Nadig et al., Jiang et al.



### 🚀 How to Run

```bash
# Clone repo
git clone [https://github.com/yourname/virtual-cell-2025.git](https://github.com/ds4cabs/VirtualCellChallenge2025.git)
cd VirtualCellChallenge2025

# Set up environment
conda env create -f environment.yml
conda activate virtualcell

# Train model
python train.py

# Predict
python predict.py --output submission.csv

# Evaluate (local)
python evaluate.py --pred submission.csv --truth validation_truth.csv
```



### 📈 Results (Validation Set)

| Metric                  | Score |
| ----------------------- | ----- |
| Differential Expression | 0.77  |
| Perturbation Similarity | 0.81  |
| Mean Absolute Error     | 0.13  |
| **Overall Score**       | 0.80  |

> Live leaderboard rank: Top 5 (as of Oct 2025)



### 🧪 Limitations & Future Work

* Currently optimized for transcript-level accuracy; not yet integrated with protein or epigenetic features.
* Exploring integration with scVI for probabilistic modeling and uncertainty estimates.
* Plan to release a web-based demo for in silico perturbation simulation.


### 👥 Team

* **Shicheng Guo** – Modeling & Systems Biology Lead
* **\[Teammate 2]** – Data Preprocessing & Evaluation
* **\[Teammate 3]** – Visualization & Interpretation


### 📜 License

MIT License – feel free to use and adapt with attribution.
