# BRACU-CSE-425-NEURAL-NETWORKS
Implementation of a non-deterministic unsupervised neural network model for clustering.


Wine Dataset Clustering with PyTorch and Scikit-Learn
--------

Overview
--------
This project explores clustering techniques on the UCI Wine dataset using
PyTorch and Scikit-Learn. It demonstrates how to load and preprocess data,
apply unsupervised clustering algorithms such as Self Organizing Map, K-Means and Gaussian Mixture
Models, and evaluate clustering performance using metrics like Silhouette Score,
Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), and stability.

The project is intended as a hands-on learning resource for unsupervised learning,
feature scaling, and evaluation of clustering algorithms.

Key Features
------------
- Data loading and preprocessing with pandas and scikit-learn
- Standardization of features using StandardScaler
- Implementation of clustering with:
  - KMeans
  - GaussianMixture
  - Self Organizing Map
- Evaluation using:
  - Silhouette Score
  - Adjusted Rand Index (ARI)
  - Normalized Mutual Information (NMI)
  - Stability
- Visualization of clustering results using Matplotlib

Setup Instructions
------------------
Launch Jupyter Notebook to explore the code.
Or, open the **CSE425_Project_Code.ipynb** file in Google Colab.

Usage
-----
Open the notebook and run all cells to:
1. Load and preprocess the Wine dataset
2. Train clustering models (KMeans, GaussianMixture, Self Organizing Map)
3. Evaluate clustering results with multiple metrics
4. Visualize performance and compare methods

Contributing
------------
Contributions, issues, and feature requests are welcome.
Feel free to fork this repo and submit pull requests.
