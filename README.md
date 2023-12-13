The purpose of this project is to examine whether different Graphical Neural Network architectures emphasize different features depending on the algorithm that is used or whether there are common important features that are emphasized regardless of what type of neural network is used. In addition, this project also aims to examine whether these important features are influenced by hyperparameters of the model (such as number of layers and channels per layer).
This project aims to examine three common GNN architectures - the GCN (Graphical Convolutional Neural Network), GAT (Graphical Attention Network), and GraphSAGE network - across two different data sets: the Planetoid Cora dataset and the Wikipedia dataset.

In this project, I first created and trained 16 different GCN, GAT, and GraphSAGE models on the Cora dataset using the same set of hyperparameter subsets.  This larger set of hyperparameters remained constant throughout our project, with changes only to the number of hidden channels per layer and the total number of layers for each model. This was due to our desire to examine how different model complexities could possibly influence which features were deemed important when predicting for model output. We then utilized the GNN Explainer algorithm outlined from the paper \textit{Ying, R., Bourgeois, D., You, J., Zitnik, M., \& Leskovec, J. (2019). Gnnexplainer: Generating explanations for graph neural networks. In Advances in Neural Information Processing Systems (pp. 9240-9251).} to generate explanations for each of the 16 models that we outlined.

The GCN and GAT models were trained transductively (meaning that I gave them both the entire graph to train on), while the GraphSAGE network was trained inductively. This was one of my reasons for choosing the GraphSAGE algorithm to examine, since I wanted to see whether transductive and inductive machine learning algorithms would respond differently to changes in model architecture.
After training these 16 models (for each type, so 48 in total), we then examined the explanations generated by the GNN Explainer algorithm. We mainly examined the 10 features deemed important for explaining the model output, and then compared (a) the consistency of these features across different models and (b) the weighted importance of the features within each model for different sets of hyperparameters. Lastly, we then examined the changes in important features across marginal changes in hyperparameters in all 3 different neural network architectures.

This was then repeated across both datasets: the CORA and Wikipedia datasets. The Cora dataset was chosen to its ubiquitous usage in the GNN literature, and I chose the WIkipedia dataset to (a) examine the effects of inductive learning on the explainability of the models and (b) to examine the effects of a larger dataset with more classes on the explainability of the models.

***How to Use***
In general, usage of this project is very simple. project.ipynb is the Jupyter notebook utilized to run the code for this project, while project.py contains the raw python script that was used. requirements.txt contains a list of all dependencies needed to properly run this project. no user input is needed to run the code, with the outputs mainly being the graphical representations of feature importance.