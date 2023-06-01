# Link Prediction on the HPO ontology using Graph Neural Networks

This repository consists of a GNN model based on Pytorch Geometric that performs link prediction in the HPO ontology and a set of genes. It uses an Heterogeneous graph dataset to train and test the model. This graph is based on the HPO version of December 2021.

The Snakefile creates a workflow that begins with the curation of the dataset, it trains a model
and generates the metric of the training and the test.

We have 3 main folders in src:

- **Analysis**: Analysis scripts. It consists of a file that defines the GNN (gnn.py), a file that creates the gnn using a given dataset and saves the generated model (create_gnn.py) and a file that generates the metrics of the test split (metrics_gnn.py). We also have a file por the explainability of the model that gives a barplot for the attributions in a given edge.

- **Data**: It consists of all the data files. We have a separated dataset for each group of gene attributes.

- **Data_preprocess**: It consists of all the scripts that generate the datasets we are using. The workflow starts downloading the HPO ontology from its website and generating a graph with the desired attributes.