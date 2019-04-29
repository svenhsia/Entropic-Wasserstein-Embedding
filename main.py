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

# # # test kl embeddings
# # g = GraphGenerator(graph_type='scale-free', n_nodes=64, m=2)
# # node_pairs = g.get_node_pairs()
# # obj_distances = g.get_obj_distances()
# # logging.info("node pairs shape: {}, obj_distances shape: {}".format(
# #     node_pairs.shape, obj_distances.shape))

# # train(node_pairs, obj_distances, embedding_type='KL', embed_dim=20, 
# #             learning_rate=0.01, nodes=64)


# # graph_id = sys.argv[1]

# # embed_dims = [2, 5, 10, 20, 30, 40]
# embed_dims = [30]
# n_epochs = 500
# num_nodes = 64

# # for graph_file in os.listdir('./graphs/'):
# #     if graph_file.split('.')[0].split('_')[-1] != graph_id:
# #         continue
# for graph_id in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11]:
#     # g = nx.read_gpickle("./graphs/{}".format(graph_file))
#     g = nx.read_gpickle("./graphs/scale_free_{}_{}.pickle".format(num_nodes, graph_id))
#     # graph_name = graph_file.split('.')[0]
#     # logging.info("Load graph {} from local file".format(graph_file))
#     graph_name = 'scale_free_{}_{}'.format(num_nodes, graph_id)
#     logging.info("Load graph {} from local file".format(graph_id))
#     node_pairs = g.get_node_pairs()
#     obj_distances = g.get_obj_distances()
#     logging.info("node pairs shape: {}, obj_distances shape: {}".format(
#         node_pairs.shape, obj_distances.shape))
    
#     for embed_dim in embed_dims:
#         # Euclidean
#         logging.info("Running Euclidean embedding, embed dim={}".format(embed_dim))
#         embeddings, loss_history, time_history, embed_distances, jac = train(
#             node_pairs, obj_distances, embedding_type='Euc', embed_dim=embed_dim, 
#             learning_rate=0.1, n_epochs=n_epochs, nodes=num_nodes)
#         np.savez('./results/{}_{}_{}'.format(graph_name, 'Euclidean', embed_dim), 
#             embeddings=embeddings, loss=loss_history, time=time_history, 
#             embed_distances=embed_distances)
        
#         # Hyperbolic
#         logging.info("Running Hyperbolic embedding, embed dim={}".format(embed_dim))
#         while True:
#             try:
#                 embeddings, loss_history, time_history, embed_distances, jac = train(
#                     node_pairs, obj_distances, embedding_type='Hyper', embed_dim=embed_dim, 
#                     learning_rate=0.01, n_epochs=n_epochs, nodes=num_nodes)
#                 break
#             except RuntimeError:
#                 logging.warning("Got Loss NaN")
#                 continue
#         np.savez('./results/{}_{}_{}'.format(graph_name, 'Hyperbolic', embed_dim), 
#             embeddings=embeddings, loss=loss_history, time=time_history, 
#             embed_distances=embed_distances)
        
#         # Wass R2
#         logging.info("Running Wasserstein R2 embedding, embed dim={}".format(embed_dim))
#         embeddings, loss_history, time_history, embed_distances, jac = train(
#             node_pairs, obj_distances, embedding_type='Wass', embed_dim=embed_dim, 
#             learning_rate=0.1, n_epochs=n_epochs, ground_dim=2, nodes=num_nodes)
#         np.savez('./results/{}_{}_{}'.format(graph_name, 'WassR2', embed_dim), 
#             embeddings=embeddings, loss=loss_history, time=time_history, 
#             embed_distances=embed_distances)
        
#         # Wass R3
#         logging.info("Running Wasserstein R3 embedding, embed dim={}".format(embed_dim))
#         embeddings, loss_history, time_history, embed_distances, jac = train(
#             node_pairs, obj_distances, embedding_type='Wass', embed_dim=embed_dim, 
#             learning_rate=0.1, n_epochs=n_epochs, ground_dim=3, nodes=num_nodes)
#         np.savez('./results/{}_{}_{}'.format(graph_name, 'WassR3', embed_dim), 
#             embeddings=embeddings, loss=loss_history, time=time_history, 
#             embed_distances=embed_distances)
        
#         # # Wass R4
#         # logging.info("Running Wasserstein R4 embedding, embed dim={}".format(embed_dim))
#         # embeddings, loss_history, time_history, embed_distances, jac = train(
#         #     node_pairs, obj_distances, embedding_type='Wass', embed_dim=embed_dim, 
#         #     learning_rate=0.1, n_epochs=n_epochs, ground_dim=4, nodes=num_nodes)
#         # np.savez('./results/{}_{}_{}'.format(graph_name, 'WassR4', embed_dim), 
#         #     embeddings=embeddings, loss=loss_history, time=time_history, 
#         #     embed_distances=embed_distances)

#         # KL
#         logging.info("Running KL embedding, embed dim={}".format(embed_dim))
#         embeddings, loss_history, time_history, embed_distances, jac = train(
#             node_pairs, obj_distances, embedding_type='KL', embed_dim=embed_dim, 
#             learning_rate=0.01, n_epochs=n_epochs, nodes=num_nodes)
#         np.savez('./results/{}_{}_{}'.format(graph_name, 'KL', embed_dim), 
#             embeddings=embeddings, loss=loss_history, time=time_history, 
#             embed_distances=embed_distances)

org_distances = np.loadtxt('./data/Sales_Transaction_Dataset.dist', delimiter=',')
logging.info("Load DTW distance data from local file")

file_name = 'Sales'

embed_dims = [30]
n_epochs = 500
num_nodes = org_distances.shape[0]
distance_adjustment = 1e-5

node_pairs = np.array([[i, j] for i in range(num_nodes) for j in range(i+1, num_nodes)])
obj_distances = np.array([org_distances[i, j] for i in range(num_nodes) for j in range(i+1, num_nodes)]) + distance_adjustment

logging.info("node pairs shape: {}, obj_distances shape: {}".format(
    node_pairs.shape, obj_distances.shape))

for embed_dim in embed_dims:

    # KL
    logging.info("Running KL embedding, embed dim={}".format(embed_dim))
    try_count = 0
    while try_count < 5:
        try:
            embeddings, loss_history, time_history, embed_distances, jac = train(
                node_pairs, obj_distances, embedding_type='KL', embed_dim=embed_dim, 
                learning_rate=0.01, n_epochs=n_epochs, nodes=num_nodes)
        except RuntimeError:
            logging.warning("Got loss NaN")
            try_count += 1
    else:
        logging.warning("Fail.") 
    np.savez('./results/{}_{}_{}'.format(file_name, 'KL', embed_dim), 
        embeddings=embeddings, loss=loss_history, time=time_history, 
        embed_distances=embed_distances)

    # Euclidean
    logging.info("Running Euclidean embedding, embed dim={}".format(embed_dim))
    try_count = 0
    while try_count < 5:
        try:
            embeddings, loss_history, time_history, embed_distances, jac = train(
                node_pairs, obj_distances, embedding_type='Euc', embed_dim=embed_dim, 
                learning_rate=0.1, n_epochs=n_epochs, nodes=num_nodes)
            break
        except RuntimeError:
            logging.warning("Got loss NaN")
            try_count += 1
    else:
        logging.warning("Fail.")
    np.savez('./results/{}_{}_{}'.format(file_name, 'Euclidean', embed_dim), 
        embeddings=embeddings, loss=loss_history, time=time_history, 
        embed_distances=embed_distances)
    
    # Hyperbolic
    logging.info("Running Hyperbolic embedding, embed dim={}".format(embed_dim))
    try_count = 0
    while try_count < 5:
        try:
            embeddings, loss_history, time_history, embed_distances, jac = train(
                node_pairs, obj_distances, embedding_type='Hyper', embed_dim=embed_dim, 
                learning_rate=0.01, n_epochs=n_epochs, nodes=num_nodes)
            break
        except RuntimeError:
            logging.warning("Got loss NaN")
            try_count += 1
    else:
        logging.warning("Fail.")
    np.savez('./results/{}_{}_{}'.format(file_name, 'Hyperbolic', embed_dim), 
        embeddings=embeddings, loss=loss_history, time=time_history, 
        embed_distances=embed_distances)
    
    # Wass R2
    logging.info("Running Wasserstein R2 embedding, embed dim={}".format(embed_dim))
    try_count = 0
    while try_count < 5:
        try:
            embeddings, loss_history, time_history, embed_distances, jac = train(
                node_pairs, obj_distances, embedding_type='Wass', embed_dim=embed_dim, 
                learning_rate=0.1, n_epochs=n_epochs, ground_dim=2, nodes=num_nodes)
            break
        except RuntimeError:
            logging.warning("Got loss NaN")
            try_count += 1
    else:
        logging.warning("Fail.")
    np.savez('./results/{}_{}_{}'.format(file_name, 'WassR2', embed_dim), 
        embeddings=embeddings, loss=loss_history, time=time_history, 
        embed_distances=embed_distances)
    
    # # Wass R3
    # logging.info("Running Wasserstein R3 embedding, embed dim={}".format(embed_dim))
    # embeddings, loss_history, time_history, embed_distances, jac = train(
    #     node_pairs, obj_distances, embedding_type='Wass', embed_dim=embed_dim, 
    #     learning_rate=0.1, n_epochs=n_epochs, ground_dim=3, nodes=num_nodes)
    # np.savez('./results/{}_{}_{}'.format(file_name, 'WassR3', embed_dim), 
    #     embeddings=embeddings, loss=loss_history, time=time_history, 
    #     embed_distances=embed_distances)
    
    # # Wass R4
    # logging.info("Running Wasserstein R4 embedding, embed dim={}".format(embed_dim))
    # embeddings, loss_history, time_history, embed_distances, jac = train(
    #     node_pairs, obj_distances, embedding_type='Wass', embed_dim=embed_dim, 
    #     learning_rate=0.1, n_epochs=n_epochs, ground_dim=4, nodes=num_nodes)
    # np.savez('./results/{}_{}_{}'.format(file_name, 'WassR4', embed_dim), 
    #     embeddings=embeddings, loss=loss_history, time=time_history, 
    #     embed_distances=embed_distances)



