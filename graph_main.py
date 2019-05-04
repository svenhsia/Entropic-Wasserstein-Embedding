import os
import sys
from time import time
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

import numpy as np
import tensorflow as tf
import networkx as nx

from graph_generator import GraphGenerator
from utils import *

# graph_id = sys.argv[1]

embed_dims = [2, 5, 10, 20, 30, 40]
n_epochs = 5
num_nodes = 64

for graph_id in range(1, 11):
    g = nx.read_gpickle("./graphs/scale_free_{}_{}.pickle".format(num_nodes, graph_id))
    graph_name = 'scale_free_{}_{}'.format(num_nodes, graph_id)
    logging.info("Load graph {} from local file".format(graph_id))
    node_pairs = g.get_node_pairs()
    obj_distances = g.get_obj_distances()
    logging.info("node pairs shape: {}, obj_distances shape: {}".format(
        node_pairs.shape, obj_distances.shape))

    batch_size = node_pairs.shape[0] # full batch
    
    for embed_dim in embed_dims:
        # Euclidean
        logging.info("Running Euclidean embedding, embed dim={}".format(embed_dim))
        embeddings, loss_history, time_history, embed_distances, jac = train(
            node_pairs, obj_distances, embedding_type='Euc', embed_dim=embed_dim, 
            learning_rate=0.1, n_epochs=n_epochs, nodes=num_nodes, batch_size=batch_size)
        np.savez('./results/{}_{}_{}'.format(graph_name, 'Euclidean', embed_dim), 
            embeddings=embeddings, loss=loss_history, time=time_history, 
            embed_distances=embed_distances)
        
        # Hyperbolic
        logging.info("Running Hyperbolic embedding, embed dim={}".format(embed_dim))
        while True:
            try:
                embeddings, loss_history, time_history, embed_distances, jac = train(
                    node_pairs, obj_distances, embedding_type='Hyper', embed_dim=embed_dim, 
                    learning_rate=0.01, n_epochs=n_epochs, nodes=num_nodes, batch_size=batch_size)
                break
            except RuntimeError:
                logging.warning("Got Loss NaN")
                continue
        np.savez('./results/{}_{}_{}'.format(graph_name, 'Hyperbolic', embed_dim), 
            embeddings=embeddings, loss=loss_history, time=time_history, 
            embed_distances=embed_distances)
        
        # Wass R2
        logging.info("Running Wasserstein R2 embedding, embed dim={}".format(embed_dim))
        embeddings, loss_history, time_history, embed_distances, jac = train(
            node_pairs, obj_distances, embedding_type='Wass', embed_dim=embed_dim, 
            learning_rate=0.1, n_epochs=n_epochs, ground_dim=2, nodes=num_nodes, batch_size=batch_size)
        np.savez('./results/{}_{}_{}'.format(graph_name, 'WassR2', embed_dim), 
            embeddings=embeddings, loss=loss_history, time=time_history, 
            embed_distances=embed_distances)
        
        # Wass R3
        logging.info("Running Wasserstein R3 embedding, embed dim={}".format(embed_dim))
        embeddings, loss_history, time_history, embed_distances, jac = train(
            node_pairs, obj_distances, embedding_type='Wass', embed_dim=embed_dim, 
            learning_rate=0.1, n_epochs=n_epochs, ground_dim=3, nodes=num_nodes, batch_size=batch_size)
        np.savez('./results/{}_{}_{}'.format(graph_name, 'WassR3', embed_dim), 
            embeddings=embeddings, loss=loss_history, time=time_history, 
            embed_distances=embed_distances)
        
        # # Wass R4
        # logging.info("Running Wasserstein R4 embedding, embed dim={}".format(embed_dim))
        # embeddings, loss_history, time_history, embed_distances, jac = train(
        #     node_pairs, obj_distances, embedding_type='Wass', embed_dim=embed_dim, 
        #     learning_rate=0.1, n_epochs=n_epochs, ground_dim=4, nodes=num_nodes, batch_size=batch_size)
        # np.savez('./results/{}_{}_{}'.format(graph_name, 'WassR4', embed_dim), 
        #     embeddings=embeddings, loss=loss_history, time=time_history, 
        #     embed_distances=embed_distances)

        # KL
        logging.info("Running KL embedding, embed dim={}".format(embed_dim))
        embeddings, loss_history, time_history, embed_distances, jac = train(
            node_pairs, obj_distances, embedding_type='KL', embed_dim=embed_dim, 
            learning_rate=0.01, n_epochs=n_epochs, nodes=num_nodes, batch_size=batch_size)
        np.savez('./results/{}_{}_{}'.format(graph_name, 'KL', embed_dim), 
            embeddings=embeddings, loss=loss_history, time=time_history, 
            embed_distances=embed_distances)

