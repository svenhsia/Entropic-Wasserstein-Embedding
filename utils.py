# import matplotlib.pyplot as plt
import tensorflow as tf
import networkx as nx

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
    r = tf.random_normal([int(u.shape[0]), 1], dtype=tf.float64)
    r_new = tf.negative(tf.ones([int(u.shape[0]), 1], dtype=tf.float64))

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


def train(node_pairs, obj_distances, embedding_type='Euc', n_epochs=500, patience=10, learning_rate=0.01, u_v=None, nodes=128, embed_dim=20, ground_dim=2, lambd=1.0, p=1, mat_bal_iter=20, mat_bal_tol=1e-5, eps=1e-5):
    if u_v is None:
        u = tf.ones([embed_dim, 1], dtype=tf.float64) / embed_dim
        v = tf.ones([embed_dim, 1], dtype=tf.float64) / embed_dim
    
    n_nodes = int(obj_distances.shape[0])

    Node_Pairs = tf.placeholder(dtype=tf.int32, shape=[n_nodes, 2], name='Node_Pairs')
    Obj_Distances = tf.placeholder(dtype=tf.float64, shape=[n_nodes], name='Obj_Distances')
    Lambd = tf.placeholder(dtype=tf.float64, shape=(), name='Lambd')
    Learning_rate = tf.placeholder(dtype=tf.float64, shape=(), name='Learning_rate')

    if (embedding_type == 'Wass'):
        Embeddings = tf.Variable(tf.random.uniform(
        [nodes, embed_dim, ground_dim], dtype=tf.float64), name='Embeddings')
        Embed_Distances = wasserstein_distances(Node_Pairs, Embeddings, u, v, Lambd, p, mat_bal_iter, mat_bal_tol)
    elif (embedding_type == 'Hyper'):
        Embeddings = tf.Variable(0.002 * tf.random.uniform([nodes, embed_dim], dtype=tf.float64) - 0.001, name='Embeddings')
        Embed_Distances = hyperbolic_distances(Node_Pairs, Embeddings, eps)
    else:
        Embeddings = tf.Variable(tf.random.uniform([nodes, embed_dim], dtype=tf.float64), name='Embeddings')
        Embed_Distances = euclidean_distances(Node_Pairs, Embeddings)

    Loss = tf.reduce_mean(tf.abs(Embed_Distances - Obj_Distances) / Obj_Distances)
    Jac = tf.gradients(ys=Embed_Distances, xs=Embeddings)
    optimizer = tf.train.AdamOptimizer(Learning_rate).minimize(Loss)
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        # Lists for storing the changing Cost and Accuracy in every Epoch
        loss_history = []

        best_loss = 1000
        early_stopping_counter = 0
        for epoch in range(n_epochs):
           # Running the Optimizer
            _, embeddings, embed_distances, loss, jac = sess.run([optimizer, Embeddings, Embed_Distances, Loss, Jac], feed_dict={Node_Pairs: node_pairs, Obj_Distances: obj_distances, Lambd: lambd, Learning_rate: learning_rate})
            # Storing loss to the history
            loss_history.append(loss)
            # Displaying result on current Epoch
            # if epoch % 10 == 0 and epoch != 0:
            print("Epoch: {}/{}, loss: {}".format(epoch, n_epochs, loss))
            # Early stopping check
            if loss < best_loss:
                best_loss = loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            if early_stopping_counter >= patience:
                break
    return embeddings, loss_history, embed_distances, jac

from graph_generator import GraphGenerator

# g = GraphGenerator(graph_type='scale-free', n_nodes=128, m=3)
g = nx.read_gpickle("./graphs/scale_free_1.pickle")
node_pairs = g.get_node_pairs()
print(node_pairs.shape)
obj_distances = g.get_obj_distances()
print(obj_distances.shape)

# learning rate = 0.1 for Euc and Wass, 0.01 for Hyper
embeddings, loss_history, embed_distances, jac = train(node_pairs, obj_distances, embedding_type='Hyper')

# plt.figure()
# plt.plot(loss_history)
# plt.show()
