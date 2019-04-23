import matplotlib.pyplot as plt
import tensorflow as tf


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


def compute_T(K, u, v, n_iter=50, tol=1e-5):
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
    r = tf.random.uniform([int(u.shape[0]), 1], dtype=tf.float64)
    c = tf.random.uniform([int(v.shape[0]), 1], dtype=tf.float64)
    r_new = tf.negative(tf.ones([int(u.shape[0]), 1], dtype=tf.float64))
    c_new = tf.negative(tf.ones([int(v.shape[0]), 1], dtype=tf.float64))

    def cond(r, c, r_new, c_new):
        r_ok = tf.reduce_all(tf.abs(r_new - r) < tol)
        c_ok = tf.reduce_all(tf.abs(c_new - c) < tol)
        return tf.logical_and(r_ok, c_ok)

    def body(r, c, r_new, c_new):
        r, c = r_new, c_new
        r_new = u / tf.matmul(K, c, False, False)
        c_new = v / tf.matmul(K, r, True, False)
        return [r, c, r_new, c_new]

    tf.while_loop(cond, body, [r, c, r_new, c_new], maximum_iterations=n_iter)

    T_opt = tf.matmul(tf.diag(tf.reshape(r, (-1,))),
                      tf.matmul(K, tf.diag(tf.reshape(c, (-1,)))))

    return T_opt


def wasserstein_distance(n1, n2, embeddings, u, v, lambd, p=1, n_iter=50, tol=1e-5):
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


def embedding_distances(pairs, embeddings, u, v, lambd, p=1, n_iter=50, tol=1e-5):

    results = tf.map_fn(lambda x: wasserstein_distance(
        x[0], x[1], embeddings, u, v, lambd, p, n_iter, tol), pairs, dtype=tf.float64)
    return results


def train(node_pairs, obj_distances, n_epochs=200, patience=10, u_v=None, embed_dim=4, ground_dim=2, lambd=0.5, p=1, mat_bal_iter=50, mat_bal_tol=1e-5):
    if u_v is None:
        u = tf.ones([embed_dim, 1], dtype=tf.float64) / embed_dim
        v = tf.ones([embed_dim, 1], dtype=tf.float64) / embed_dim
    
    n_nodes = int(obj_distances.shape[0])

    Node_Pairs = tf.placeholder(dtype=tf.int32, shape=[None, 2], name='Node_Pairs')
    Obj_Distances = tf.placeholder(dtype=tf.float64, shape=[None], name='Obj_Distances')

    Embeddings = tf.Variable(tf.random.uniform(
        [n_nodes, embed_dim, ground_dim], dtype=tf.float64), name='Embeddings')
    Embed_Distances = embedding_distances(Node_Pairs, Embeddings, u, v, lambd, p, mat_bal_iter, mat_bal_tol)
    Loss = tf.reduce_mean(tf.abs(Embed_Distances - Obj_Distances) / Obj_Distances)
    Jac = tf.gradients(ys=Embed_Distances, xs=Embeddings)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(Loss)
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        # Lists for storing the changing Cost and Accuracy in every Epoch
        loss_history = []

        best_loss = 1000
        early_stopping_counter = 0
        for epoch in range(n_epochs):
           # Running the Optimizer
            _, embeddings, loss, embed_distances, jac = sess.run([optimizer, Embeddings, Embed_Distances, Loss, Jac], feed_dict={Node_Pairs: node_pairs, Obj_Distances: obj_distances})
            # Storing loss to the history
            loss_history.append(loss)
            # Displaying result on current Epoch
            if epoch % 10 == 0 and epoch != 0:
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

g = GraphGenerator(graph_type='scale-free')
node_pairs = g.get_node_pairs()
obj_distances = g.get_obj_distances()

embeddings, loss_history, embed_distances, jac  = train(node_pairs, obj_distances)

plt.figure()
plt.plot(loss_history)
plt.show()
