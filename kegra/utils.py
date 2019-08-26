from __future__ import print_function

import scipy.sparse as sp
import numpy as np
from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def load_dataset_comm(file_name):
    """Load a graph from a Numpy binary file.

    Parameters
    ----------
    file_name : str
        Name of the file to load.

    Returns
    -------
    graph : dict
        Dictionary that contains:
            * 'A' : The adjacency matrix in sparse matrix format
            * 'X' : The attribute matrix in sparse matrix format
            * 'F' : The community labels in sparse matrix format
            * Further dictionaries mapping node, class and attribute IDs

    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        A = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                           loader['adj_indptr']), shape=loader['adj_shape'])

        X = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                           loader['attr_indptr']), shape=loader['attr_shape'])

        F = sp.csr_matrix((loader['labels_data'], loader['labels_indices'],
                           loader['labels_indptr']), shape=loader['labels_shape'])

        graph = {
            'A': A,
            'X': X,
            'F': F
        }

        node_names = loader.get('node_names')
        if node_names is not None:
            node_names = node_names.tolist()
            graph['node_names'] = node_names

        attr_names = loader.get('attr_names')
        if attr_names is not None:
            attr_names = attr_names.tolist()
            graph['attr_names'] = attr_names

        class_names = loader.get('class_names')
        if class_names is not None:
            class_names = class_names.tolist()
            graph['class_names'] = class_names

        return graph

def load_data(path="data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))

    return features.todense(), adj, labels


def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm


def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj, symmetric)
    return adj


def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def get_splits(y):
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)
    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)
    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    y_test[idx_test] = y[idx_test]
    train_mask = sample_mask(idx_train, y.shape[0])
    return y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask


def categorical_crossentropy(preds, labels):
    return np.mean(-np.log(np.extract(labels, preds)))


def accuracy(preds, labels):
    return np.mean(np.equal(np.argmax(labels, 1), np.argmax(preds, 1)))


def evaluate_preds(preds, labels, indices):

    split_loss = list()
    split_acc = list()

    for y_split, idx_split in zip(labels, indices):
        split_loss.append(categorical_crossentropy(preds[idx_split], y_split[idx_split]))
        split_acc.append(accuracy(preds[idx_split], y_split[idx_split]))

    return split_loss, split_acc


def normalized_laplacian(adj, symmetric=True):
    adj_normalized = normalize_adj(adj, symmetric)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    return laplacian


def rescale_laplacian(laplacian):
    try:
        print('Calculating largest eigenvalue of normalized graph Laplacian...')
        largest_eigval = eigsh(laplacian, 1, which='LM', return_eigenvectors=False)[0]
    except ArpackNoConvergence:
        print('Eigenvalue calculation did not converge! Using largest_eigval=2 instead.')
        largest_eigval = 2

    scaled_laplacian = (2. / largest_eigval) * laplacian - sp.eye(laplacian.shape[0])
    return scaled_laplacian


def chebyshev_polynomial(X, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    T_k = list()
    T_k.append(sp.eye(X.shape[0]).tocsr())
    T_k.append(X)

    def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X):
        X_ = sp.csr_matrix(X, copy=True)
        return 2 * X_.dot(T_k_minus_one) - T_k_minus_two

    for i in range(2, k+1):
        T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2], X))

    return T_k


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

"""
Loss functions for overlapping community detection.
"""
import numpy as np
import scipy.sparse as sp
import tensorflow as tf


def get_uniform_edge_sampler(A, num_pos, num_neg):
    """Return a generator for edges and non-edges from the given graph.

    The output of the generator may contain duplicates.


    Parameters
    ----------
    A : sp.spmatrix
        Adajcency matrix of the graph.
    num_pos : int
        Number of edges to return.
    num_neg : int
        Number of non-edges to return.

    Returns
    -------
    next_batch : generator
        Generator that returns (next_edges, next_nonedges) tuple at each call.

    """
    if (A != A.T).sum():
        raise ValueError("Adjacency matrix must be undirected.")
    edges = np.column_stack(sp.triu(A).nonzero())
    num_nodes = A.shape[0]
    num_edges = edges.shape[0]
    def next_batch():
        while True:
            # Select num_pos edges
            edges_idx = np.random.randint(0, num_edges, size=num_pos)
            next_edges = edges[edges_idx, :]

            # Select num_neg non-edges
            generated = False
            while not generated:
                candidate_ne = np.random.randint(0, num_nodes, size=(2*num_neg, 2), dtype=np.int32)
                cne1, cne2 = candidate_ne[:, 0], candidate_ne[:, 1]
                to_keep = (1 - A[cne1, cne2]).astype(np.bool).A1 * (cne1 != cne2)
                next_nonedges = candidate_ne[to_keep][:num_neg]
                generated = to_keep.sum() >= num_neg
            yield next_edges, next_nonedges
    return next_batch


def berpo_loss(F, A, p_no_comm=1e-4, neg_scale=1, stochastic=False, batch_size=10000):
    """Evaluate the loss (negative log-likelihood) under the Bernoulli-Poisson model.

    Parameters
    ----------
    F : tf.Tensor, shape [N, K]
        Non-negative community memberships for each node.
    A : np.ndarray or sp.csr_matrix, shape [N, N]
        Binary symmetric adjacency matrix.
    p_no_comm : float, optional
        Edge probability between nodes that share no communities.
    neg_scale : float, optional
        Weighting factor for negative examples.
        neg_scale = 1 corresponds to balanced loss.
        neg_scale = 'auto' sets it to num_nonedges / num_edges.
        This corresponds to the vanilla loss.

    Returns
    -------
    loss : float
        Value of the loss (negative log-likelihood), lower is better.

    """
    N = A.shape[0]
    if neg_scale == 'auto':
        neg_scale = (N**2 - N - A.nnz) / A.nnz
        print(f'Automatically setting neg_scale = {neg_scale}')
    if stochastic:
        next_batch = get_uniform_edge_sampler(A, batch_size, batch_size)
        dataset = tf.data.Dataset.from_generator(next_batch, (tf.int32, tf.int32), ((None), (None)))
        train_ones, train_zeros = dataset.prefetch(1).make_one_shot_iterator().get_next()
        return minibatch_berpo_loss(F, train_ones, train_zeros, p_no_comm, neg_scale=neg_scale)
    else:
        return full_berpo_loss(F, A, p_no_comm=p_no_comm, neg_scale=neg_scale)


def minibatch_berpo_loss(F, edges_idx, nonedges_idx, p_no_comm=1e-4, neg_scale=1):
    """Compute negative log-likelihood of given edges and nonedges under the BerPo model.

    Parameters
    ----------
    F : tf.Tensor, shape [N, K]
        Non-negative community memberships for each node.
    edges_idx : tf.Tensor, shape [num_edges, 2]
        List of edges, each row representing the (u, v) tuple.
    nonedges_idx : tf.Tensor, shape [num_nonedges, 2]
        List of non-edges, each row representing the (u, v) tuple.
    p_no_comm : float, optional
        Edge probability between nodes that share no communities.
    neg_scale : float, optional
        Weighting factor for negative examples.
        neg_scale = 1 corresponds to balanced loss.
        neg_scale = num_nonedges / num_edges corresponds to vanilla loss.

    Returns
    -------
    loss : tf.Tensor, shape []
        Value of the loss (mean NLL of edges + mean NLL of non-edges).

    """
    e1, e2 = edges_idx[:, 0], edges_idx[:, 1]
    edge_dots = tf.reduce_sum(tf.gather(F, e1) * tf.gather(F, e2), axis=1)
    eps = np.log(1 / (1 - p_no_comm))  # probability of edge if no communities are shared
    loss_edges = -tf.reduce_mean(tf.log1p(-tf.exp(-eps - edge_dots)))

    ne1, ne2 = nonedges_idx[:, 0], nonedges_idx[:, 1]
    loss_nonedges = tf.reduce_mean(tf.reduce_sum(tf.gather(F, ne1) * tf.gather(F, ne2), axis=1))
    return (loss_edges + neg_scale * loss_nonedges) / (1 + neg_scale)


def full_berpo_loss(F, A, p_no_comm=1e-4, neg_scale=1.0):
    """Compute full BerPo loss in Tensorflow.

    Parameters
    ----------
    F : np.ndarray or tf.Tensor, shape [N, K]
        Nonnegative community affiliation matrix.
    A : np.ndarray or sp.csr_matrix, shape [N, N]
        Binary symmetric adjacency matrix.
    p_no_comm : float, optional
        Edge probability between nodes that share no communities.
    neg_scale : float, optional
        Weighting factor for negative examples.
        neg_scale = 1 corresponds to balanced loss.
        neg_scale = num_nonedges / num_edges corresponds to vanilla loss.

    Returns
    -------
    loss : float
        Value of the loss (negative log-likelihood), lower is better.

    """
    e1, e2 = A.nonzero()
    edge_dots = tf.reduce_sum(tf.gather(F, e1) * tf.gather(F, e2), axis=1)
    eps = np.log(1 / (1 - p_no_comm))  # probability of edge if no communities are shared
    loss_edges = -tf.reduce_sum(tf.log1p(-tf.exp(-eps - edge_dots)))

    # Correct for overcounting F_u * F_v for edges and nodes with themselves
    self_dots_sum = tf.reduce_sum(F * F)
    correction = self_dots_sum + tf.reduce_sum(edge_dots)
    sum_F = tf.transpose(tf.reduce_sum(F, axis=0, keepdims=True))
    loss_nonedges = tf.reduce_sum(F @ sum_F) - correction

    N = A.shape[0]
    num_edges = A.nnz
    num_nonedges = N**2 - N - num_edges
    return (loss_edges / num_edges + neg_scale * loss_nonedges / num_nonedges) / (1 + neg_scale)
