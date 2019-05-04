import os
import sys
from time import time
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

import numpy as np
import tensorflow as tf

from utils import *

org_distances = np.loadtxt('./data/Sales_Transaction_Dataset.dist', delimiter=',')
logging.info("Load DTW distance data from local file")

file_name = 'Sales'

embed_dims = [30]
n_epochs = 30
batch_size = 4096
num_nodes = org_distances.shape[0]
distance_adjustment = 1e-5

node_pairs = np.array([[i, j] for i in range(num_nodes) for j in range(i+1, num_nodes)])
obj_distances_origin = np.array([org_distances[i, j] for i in range(num_nodes) for j in range(i+1, num_nodes)])

logging.info("node pairs shape: {}, obj_distances shape: {}".format(
    node_pairs.shape, obj_distances_origin.shape))

max_try = 1
normalize_distance = True

if normalize_distance:
    obj_min = obj_distances_origin.min()
    obj_max = obj_distances_origin.max()
    obj_distances = (obj_distances_origin - obj_min) / (obj_max - obj_min) + distance_adjustment
else:
    obj_distances = obj_distances_origin + distance_adjustment

for embed_dim in embed_dims:

    # Wass R2
    logging.info("Running Wasserstein R2 embedding, embed dim={}".format(embed_dim))
    try_count = 0
    while try_count < max_try:
        try:
            embeddings, loss_history, time_history, embed_distances, jac = train(
                node_pairs, obj_distances, embedding_type='Wass', embed_dim=embed_dim, 
                learning_rate=0.001, n_epochs=n_epochs, ground_dim=2, nodes=num_nodes, batch_size=batch_size)
            break
        except RuntimeError:
            logging.warning("Got loss NaN")
            try_count += 1
    else:
        logging.warning("Fail.")
    if normalize_distance:
        embed_distances = (embed_distances - distance_adjustment) * (obj_max - obj_min) + obj_min
    logging.info("Writing {}_{}_{}_batch to local file".format(file_name, 'WassR2', embed_dim))
    np.savez('./results/{}_{}_{}_batch'.format(file_name, 'WassR2', embed_dim), 
        embeddings=embeddings, loss=loss_history, time=time_history, 
        embed_distances=embed_distances)

    # KL
    logging.info("Running KL embedding, embed dim={}".format(embed_dim))
    try_count = 0
    while try_count < max_try:
        try:
            embeddings, loss_history, time_history, embed_distances, jac = train(
                node_pairs, obj_distances, embedding_type='KL', embed_dim=embed_dim, 
                learning_rate=0.01, n_epochs=n_epochs, nodes=num_nodes, batch_size=batch_size)
            break
        except RuntimeError:
            logging.warning("Got loss NaN")
            try_count += 1
    else:
        logging.warning("Fail.")
    if normalize_distance:
        embed_distances = (embed_distances - distance_adjustment) * (obj_max - obj_min) + obj_min
    logging.info("Writing {}_{}_{}_batch to local file".format(file_name, 'KL', embed_dim))
    np.savez('./results/{}_{}_{}_batch'.format(file_name, 'KL', embed_dim), 
        embeddings=embeddings, loss=loss_history, time=time_history, 
        embed_distances=embed_distances)

    # Euclidean
    logging.info("Running Euclidean embedding, embed dim={}".format(embed_dim))
    try_count = 0
    while try_count < max_try:
        try:
            embeddings, loss_history, time_history, embed_distances, jac = train(
                node_pairs, obj_distances, embedding_type='Euc', embed_dim=embed_dim, 
                learning_rate=0.001, n_epochs=n_epochs, nodes=num_nodes, batch_size=batch_size)
            break
        except RuntimeError:
            logging.warning("Got loss NaN")
            try_count += 1
    else:
        logging.warning("Fail.")
    if normalize_distance:
        embed_distances = (embed_distances - distance_adjustment) * (obj_max - obj_min) + obj_min
    logging.info("Writing {}_{}_{}_batch to local file".format(file_name, 'Euclidean', embed_dim))
    np.savez('./results/{}_{}_{}_batch'.format(file_name, 'Euclidean', embed_dim), 
        embeddings=embeddings, loss=loss_history, time=time_history, 
        embed_distances=embed_distances)
    
    # Hyperbolic
    logging.info("Running Hyperbolic embedding, embed dim={}".format(embed_dim))
    try_count = 0
    while try_count < max_try:
        try:
            embeddings, loss_history, time_history, embed_distances, jac = train(
                node_pairs, obj_distances, embedding_type='Hyper', embed_dim=embed_dim, 
                learning_rate=0.00005, n_epochs=n_epochs, nodes=num_nodes, batch_size=batch_size)
            break
        except RuntimeError:
            logging.warning("Got loss NaN")
            try_count += 1
    else:
        logging.warning("Fail.")
    if normalize_distance:
        embed_distances = (embed_distances - distance_adjustment) * (obj_max - obj_min) + obj_min
    logging.info("Writing {}_{}_{}_batch to local file".format(file_name, 'Hyperbolic', embed_dim))
    np.savez('./results/{}_{}_{}_batch'.format(file_name, 'Hyperbolic', embed_dim), 
        embeddings=embeddings, loss=loss_history, time=time_history, 
        embed_distances=embed_distances)
    
    
    # # Wass R3
    # logging.info("Running Wasserstein R3 embedding, embed dim={}".format(embed_dim))
    # embeddings, loss_history, time_history, embed_distances, jac = train(
    #     node_pairs, obj_distances, embedding_type='Wass', embed_dim=embed_dim, 
    #     learning_rate=0.001, n_epochs=n_epochs, ground_dim=3, nodes=num_nodes, batch_size=batch_size)
    # np.savez('./results/{}_{}_{}'.format(file_name, 'WassR3', embed_dim), 
    #     embeddings=embeddings, loss=loss_history, time=time_history, 
    #     embed_distances=embed_distances)
    
    # # Wass R4
    # logging.info("Running Wasserstein R4 embedding, embed dim={}".format(embed_dim))
    # embeddings, loss_history, time_history, embed_distances, jac = train(
    #     node_pairs, obj_distances, embedding_type='Wass', embed_dim=embed_dim, 
    #     learning_rate=0.001, n_epochs=n_epochs, ground_dim=4, nodes=num_nodes, batch_size=batch_size)
    # np.savez('./results/{}_{}_{}'.format(file_name, 'WassR4', embed_dim), 
    #     embeddings=embeddings, loss=loss_history, time=time_history, 
    #     embed_distances=embed_distances)



