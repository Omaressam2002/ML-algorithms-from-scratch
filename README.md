# 🧠 Machine Learning Algorithms from Scratch

This repository contains **clean and well-documented implementations of core Machine Learning algorithms built entirely from scratch** using only fundamental Python libraries such as `NumPy`, `Matplotlib`, and `Pandas`.  

The purpose of this project is to **deeply understand how popular ML algorithms work under the hood** — from linear models to ensemble methods and dimensionality reduction — without relying on frameworks like scikit-learn or TensorFlow.

---

## 🚀 Implemented Algorithms

### 📈 **Regression Models**
- **Linear Regression** — implements gradient descent optimization and performance evaluation using MSE and R² metrics.
- **Logistic Regression** — binary and multi-class classification with sigmoid activation and cross-entropy loss.

### 🧮 **Probabilistic Models**
- **Naive Bayes Classifier** — Gaussian and Multinomial versions with probabilistic predictions and text classification examples.

### 🌳 **Tree-Based Models**
- **Decision Tree Classifier** — built from scratch using entropy and Gini impurity; includes visualization and interpretability tools.
- **Bagging (Bootstrap Aggregation)** — ensemble of decision trees to reduce variance.
- **Boosting (AdaBoost)** — sequential tree boosting with adaptive sample weighting.

### 🧑‍🤝‍🧑 **Instance-Based Learning**
- **K-Nearest Neighbors (KNN)** — distance-based classifier with visualization of Voronoi regions and decision boundaries.

### 🔵 **Clustering & Dimensionality Reduction**
- **K-Means Clustering** — iterative centroid-based clustering with random initialization and convergence plots.
- **Principal Component Analysis (PCA)** — dimensionality reduction and feature decorrelation from scratch, with eigen decomposition.

### ⚔️ **Support Vector Machines**
- **SVM (Support Vector Machine)** — linear classifier using hinge loss and margin maximization principles.
- **SVR (Support Vector Regression)** — regression version with ε-insensitive loss and evaluation on continuous datasets (California Housing).

---

## 🧩 Datasets Used

A variety of datasets were used to demonstrate and test each algorithm:

| Category | Dataset | Description |
|-----------|----------|-------------|
| Classification | **AG News**, **Iris**, **Banknote Authentication**, **Titanic** | Text, binary, and multi-class classification tasks |
| Regression | **California Housing** | Continuous prediction task for median house values |
| Clustering | **Synthetic Gaussian blobs**, **PCA-projected data** | Used for unsupervised learning demos |

---

## 🧠 Key Features

- Implemented **from scratch** (no ML frameworks)
- Uses only **NumPy**, **Pandas**, and **Matplotlib**
- Includes **data preprocessing**, **evaluation metrics**, and **visualizations**
- Designed for **learning and educational purposes**
- Clean, **object-oriented class-based design**
- Tested and visualized using **real-world datasets**

---

## 📊 Example Visualizations

- Confusion matrices for classifiers  
- Decision boundaries for SVM, KNN, and logistic regression  
- PCA 2D projections and explained variance plots  
- K-Means cluster visualizations and centroid tracking  
- Regression scatterplots and performance trends  

---

## 🧰 Tech Stack

| Tool | Purpose |
|------|----------|
| **Python** | Core programming language |
| **NumPy** | Numerical computations |
| **Pandas** | Data handling |
| **Matplotlib / Seaborn** | Visualization |
| **Scikit-learn (for datasets only)** | To load and benchmark models |


