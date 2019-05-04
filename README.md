# Entropic-Wasserstein-Embedding
This is a course project of « Geometric Methods in Machine Learning » @ ENSAE, which aims at implementing the paper « Learning Embeddings into Wasserstein Spaces » (C. Frogner, et al.)

# Prerequisite
1. Install ```networkx (v2.2)```, ```tensorflow```.
2. Create ```graphs```, ```data``` and ```results``` folders.

# How to execute the code
1. To get results on scale-free networks, first decomment ```line 67-77``` in ```graph_generator.py```. Then run ```graph_generator.py``` and you will see 10 generated networks under ```graphs``` folder.
2. Run ```graph_main.py``` to get embedding results under ```results``` folder. This will take several hours to run.
3. To get results on __Sales Transaction Weekly Dataset__, first download data from https://archive.ics.uci.edu/ml/datasets/Sales_Transactions_Dataset_Weekly and save it to ```data``` folder.
4. Run ```DTWdistance.py``` and you will see the calculated DTW distances are stored as ```data/Sales_Transactions_Dataset.dist```. This will take around 20 minutes to run.
5. Run ```sales_main.py``` and you will see the embedding results are stored under ```results``` folder. This will take around one hour to run.
6. The visualization results are in ```Visualization.ipynb```.
