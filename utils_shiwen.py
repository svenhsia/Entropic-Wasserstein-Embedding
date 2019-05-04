import os
import sys
from time import time
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

import numpy as np
import tensorflow as tf
import networkx as nx
# import matplotlib.pyplot as plt

def cdist(X, Y):
    X2 = tf.reduce_sum(tf.square(X), 1)
    Y2 = tf.reduce_sum(tf.square(Y), 1)
    X2 = tf.reshape(X2, [-1, 1])
    Y2 = tf.reshape(Y2, [1, -1])
    # return pairwise euclidead difference matrix
    distances = tf.sqrt(tf.maximum(
        X2 + Y2 - 2 * tf.matmul(X, Y, False, True), 0.0))
    assert distances.shape == [X.shape[0], Y.shape[0]]
    return distances


def compute_T(K, u, v, n_iter, tol):
    """
    Parameters:
    -----------
        D: 2D array, [M, N]
        r: 1D array, [M, ]
        c: 1D array, [N, ]
        lambd: regularization parameter in Sinkhorn divergence
        p: power of the Wasserstein space
        n_iter: number of iterations for matrix balancing
        tol: tolerance for stopping matrix balancing iterations
    """
    K_tilde = 1. / u * K
    r = tf.zeros([int(u.shape[0]), 1], dtype=tf.float64)
    r_new = tf.ones([int(u.shape[0]), 1], dtype=tf.float64)

    def cond(r, r_new):
        r_enter = tf.reduce_any(tf.abs(r_new - r) > tol)
        return r_enter

    def body(r, r_new):
        r = r_new
        r_new = 1. / tf.matmul(K_tilde, v / tf.matmul(K, r, True, False))
        return [r, r_new]

    _, r = tf.while_loop(cond, body, [r, r_new], maximum_iterations=n_iter)
    c = v / tf.matmul(K, r, True, False)

    T_opt = tf.matmul(tf.diag(tf.reshape(r, (-1,))),
                      tf.matmul(K, tf.diag(tf.reshape(c, (-1,)))))

    return T_opt


def wasserstein_distance(n1, n2, embeddings, u, v, lambd, p, n_iter, tol):
    support_1 = embeddings[n1, :, :]
    support_2 = embeddings[n2, :, :]
    D = cdist(support_1, support_2)
    D_p = tf.pow(D, p)
    K = tf.exp(-D_p / lambd)
    T = compute_T(K, u, v, n_iter, tol)

    # distance = tf.trace(tf.matmul(D_p, T, False, True)) + lambd * \
    #     tf.trace(tf.matmul(T, tf.log(T) -
    #                        tf.ones(T.shape, dtype=tf.float64), False, True))
    distance = tf.trace(tf.matmul(D_p, T, False, True))
    return distance


def wasserstein_distances(pairs, embeddings, u, v, lambd, p, n_iter, tol):
    results = tf.map_fn(lambda x: wasserstein_distance(
        x[0], x[1], embeddings, u, v, lambd, p, n_iter, tol), pairs, dtype=tf.float64)
    return results

def euclidean_distance(n1, n2, embeddings):
    v1 = embeddings[n1, :]
    v2 = embeddings[n2, :]
    distance = tf.sqrt(tf.reduce_sum(tf.square(v1 - v2)))
    return distance

def euclidean_distances(pairs, embeddings):
    results = tf.map_fn(lambda x: euclidean_distance(x[0], x[1], embeddings), pairs, dtype=tf.float64)
    return results

def hyperbolic_distance(n1, n2, embeddings, eps):
    v1 = embeddings[n1, :]
    v2 = embeddings[n2, :]
    norm1 = tf.norm(v1)
    norm2 = tf.norm(v2)
    v1 = tf.cond(tf.greater_equal(norm1, 1), lambda: v1 / norm1 - eps, lambda: v1)
    v2 = tf.cond(tf.greater_equal(norm2, 1), lambda: v2 / norm2 - eps, lambda: v2)
    
    distance = tf.acosh(1 + 2 * tf.reduce_sum(tf.square(v1 - v2)) / ((1 - tf.reduce_sum(tf.square(v1))) * (1 - tf.reduce_sum(tf.square(v2)))))
    return distance

def hyperbolic_distances(pairs, embeddings, eps):
    results = tf.map_fn(lambda x: hyperbolic_distance(x[0], x[1], embeddings, eps), pairs, dtype=tf.float64)
    return results

def kl_distance(n1, n2, embeddings, eps):
    v1 = embeddings[n1, :]
    v2 = embeddings[n2, :]
    min1 = tf.reduce_min(v1)
    min2 = tf.reduce_min(v2)
    v1 = tf.cond(tf.less_equal(min1, 0), lambda: v1 - min1 + eps, lambda: v1)
    v2 = tf.cond(tf.less_equal(min2, 0), lambda: v2 - min2 + eps, lambda: v2)
    v1 = v1 / tf.norm(v1)
    v2 = v2 / tf.norm(v2)
    kl = (tf.reduce_sum(v1 * (tf.log(v1) - tf.log(v2))) + tf.reduce_sum(v2 * (tf.log(v2) - tf.log(v1)))) / 2
    return kl

def kl_distances(pairs, embeddings, eps):
    results = tf.map_fn(lambda x: kl_distance(x[0], x[1], embeddings, eps), pairs, dtype=tf.float64)
    return results


def train(node_pairs, obj_distances, embedding_type='Euc', n_epochs=500, batch_size=32, learning_rate=0.01, u_v=None, nodes=128, embed_dim=20, ground_dim=2, lambd=1.0, p=1, mat_bal_iter=20, mat_bal_tol=1e-5, eps=1e-5):
    if u_v is None:
        u = tf.ones([embed_dim, 1], dtype=tf.float64) / embed_dim
        v = tf.ones([embed_dim, 1], dtype=tf.float64) / embed_dim
    
    n_pairs = int(obj_distances.shape[0])
    node_pairs_idx = np.arange(n_pairs)
    n_batches = n_pairs // batch_size

    Node_Pairs = tf.placeholder(dtype=tf.int32, shape=[None, 2], name='Node_Pairs')
    Obj_Distances = tf.placeholder(dtype=tf.float64, shape=[None], name='Obj_Distances')
    Lambd = tf.placeholder(dtype=tf.float64, shape=(), name='Lambd')
    Learning_rate = tf.placeholder(dtype=tf.float64, shape=(), name='Learning_rate')

    if embedding_type == 'Wass':
        Embeddings = tf.Variable(tf.random.uniform(
        [nodes, embed_dim, ground_dim], dtype=tf.float64), name='Embeddings')
        Embed_Distances = wasserstein_distances(Node_Pairs, Embeddings, u, v, Lambd, p, mat_bal_iter, mat_bal_tol)
    elif embedding_type == 'Hyper':
        Embeddings = tf.Variable(0.002 * tf.random.uniform([nodes, embed_dim], dtype=tf.float64) - 0.001, name='Embeddings')
        Embed_Distances = hyperbolic_distances(Node_Pairs, Embeddings, eps)
    elif embedding_type == 'KL':
        Embeddings = tf.Variable(tf.random.uniform([nodes, embed_dim], dtype=tf.float64), name='Embeddings')
        Embed_Distances = kl_distances(Node_Pairs, Embeddings, eps)
    else:
        Embeddings = tf.Variable(tf.random.uniform([nodes, embed_dim], dtype=tf.float64), name='Embeddings')
        Embed_Distances = euclidean_distances(Node_Pairs, Embeddings)

    Loss = tf.reduce_mean(tf.abs(Embed_Distances - Obj_Distances) / Obj_Distances)
    # Loss = tf.reduce_mean(tf.square(Embed_Distances - Obj_Distances))
    Jac = tf.gradients(ys=Embed_Distances, xs=Embeddings)
    optimizer = tf.train.AdamOptimizer(Learning_rate).minimize(Loss)
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        # Lists for storing the changing Cost and Accuracy in every Epoch
        loss_history = []
        time_history = []

        # best_loss = 1000
        # early_stopping_counter = 0
        node_pairs_idx = np.arange(len(node_pairs))
        start_time = time()
        for epoch in range(n_epochs):
           # Running the Optimizer
            np.random.shuffle(node_pairs_idx)
            for batch_id in range(n_batches):
                batch_node_pairs_idx = node_pairs_idx[batch_id*batch_size:(batch_id+1)*batch_size]
                batch_node_pairs = node_pairs[batch_node_pairs_idx, :]
                batch_obj_distances = obj_distances[batch_node_pairs_idx]
                _, embeddings, embed_distances, loss, jac = sess.run([optimizer, Embeddings, Embed_Distances, Loss, Jac], 
                                                                     feed_dict={Node_Pairs: batch_node_pairs, 
                                                                                Obj_Distances: batch_obj_distances, 
                                                                                Lambd: lambd, 
                                                                                Learning_rate: learning_rate})
                if np.isnan(loss):
                    raise RuntimeError("Loss is NaN.")
                if batch_id % (n_batches // 10) == 0:
                    logging.info("Epoch: {}/{}, batch No.{}/{} loss: {}".format(epoch+1, n_epochs, batch_id+1, n_batches, loss))
            loss = sess.run(Loss, feed_dict={Node_Pairs: node_pairs, Obj_Distances: obj_distances, Lambd: lambd, Learning_rate: learning_rate})
            # Storing loss to the history
            loss_history.append(loss)
            # Storing consumed time
            time_history.append(time() - start_time)
            # Displaying result on current Epoch
            if epoch % 1 == 0:
                logging.info("Epoch: {}/{}, loss: {}".format(epoch+1, n_epochs, loss))
            # Early stopping check
            if epoch > 10 and np.mean(loss_history[-6:-1]) - loss_history[-1] < 1e-4:
                logging.info("Early Stopped: 10 consecutive epochs with loss improvement {}".format(loss_history[-2]-loss_history[-1]))
                break
            # if loss < best_loss:
            #     best_loss = loss
            #     early_stopping_counter = 0
            # else:
            #     early_stopping_counter += 1
            # if early_stopping_counter >= patience:
            #     break
    return embeddings, loss_history, time_history, embed_distances, jac

