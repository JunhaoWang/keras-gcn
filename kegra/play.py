from __future__ import print_function
import tensorflow as tf
tf.enable_eager_execution()

import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers

import tensorflow.keras.layers as tkl
from keras.regularizers import l2
from layers.graph import GraphConvolution
from utils import *
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Define parameters
DATASET = 'cora'
FILTER = 'localpool'  # 'chebyshev'
MAX_DEGREE = 2  # maximum polynomial degree
SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
NB_EPOCH = 2000
PATIENCE = 10  # early stopping patience


# Get data
X, A, y = load_data(dataset=DATASET)
y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits(y)

pos_weight = float(A.shape[0] * A.shape[0] - A.sum()) / A.sum()
norm = A.shape[0] * A.shape[0] / float((A.shape[0] * A.shape[0] - A.sum()) * 2)
labels = np.argmax(y, 1)
num_nodes = len(y)

# Normalize X
X /= X.sum(1).reshape(-1, 1)

if FILTER == 'localpool':
    """ Local pooling filters (see 'renormalization trick' in Kipf & Welling, arXiv 2016) """
    print('Using local pooling filters...')
    A_ = preprocess_adj(A, SYM_NORM)
    support = 1
    graph = [X, A_]
    # G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True)]

elif FILTER == 'chebyshev':
    """ Chebyshev polynomial basis filters (Defferard et al., NIPS 2016)  """
    print('Using Chebyshev polynomial basis filters...')
    L = normalized_laplacian(A, SYM_NORM)
    L_scaled = rescale_laplacian(L)
    T_k = chebyshev_polynomial(L_scaled, MAX_DEGREE)
    support = MAX_DEGREE + 1
    graph = [X]+T_k
    # G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True) for _ in range(support)]

else:
    raise Exception('Invalid filter type.')

def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)

def recon_edge_helper(latent, method='dot'):
    if not tf.is_tensor(latent):
        latent = latent.sample()
    if method == 'dot':
        result = tf.reshape(tf.matmul(latent, tf.transpose(latent)), [-1])
    return result

def _softplus_inverse(x):
  """Helper which computes the function inverse of `tf.nn.softplus`."""
  return tf.math.log(tf.math.expm1(x))

def visualzie_embeddding(embeddings, labels, name):
    if tf.is_tensor(embeddings):
        embeddings = embeddings.numpy()
    else:
        embeddings = embeddings.sample().numpy()
    X_embedded = PCA().fit_transform(embeddings)
    for l in set(labels):
        plt.scatter(X_embedded[np.argwhere(labels == l), 0], X_embedded[np.argwhere(labels == l), 1], label=l)
    plt.legend()
    # plt.show()
    plt.savefig(name)
    plt.clf()
    plt.close()


class GAE(tf.keras.Model):
    def __init__(self):
        super(GAE, self).__init__()
        self.conv1 = GraphConvolution(
            16, 1 , activation='relu', kernel_regularizer=l2(5e-4), name='conv1'
        )
        self.conv2 = GraphConvolution(
            7, 1, name='conv2'
        )

    def call(self, inputs):
        X = tf.cast(tf.convert_to_tensor(inputs[0]), tf.float32)
        G = tf.cast(convert_sparse_matrix_to_sparse_tensor(inputs[1]), tf.float32)
        inputs_tensor = [X, G]
        result = self.conv1(inputs_tensor)
        result = self.conv2([result, G])
        return result

    def recon_edge(self, inputs):
        """Run the model."""
        latent = self.call(inputs)
        return recon_edge_helper(latent)

class VGAE(tf.keras.Model):
    def __init__(self):
        super(VGAE, self).__init__()
        self.conv1 = GraphConvolution(
            16, 1 , activation='relu', kernel_regularizer=l2(5e-4), name='conv1'
        )
        self.conv2 = GraphConvolution(
            7, 1, name='conv2'
        )
        self.conv3 = GraphConvolution(
            7, 1, name='conv2'
        )


    def call(self, inputs):
        X = tf.cast(tf.convert_to_tensor(inputs[0]), tf.float32)
        G = tf.cast(convert_sparse_matrix_to_sparse_tensor(inputs[1]), tf.float32)
        inputs_tensor = [X, G]
        latent = self.conv1(inputs_tensor)
        self.z_mean = self.conv2([latent, G])
        self.z_log_std = self.conv3([latent, G])
        z_sample = self.z_mean + tf.random_normal(self.z_mean.shape.as_list()) * tf.exp(self.z_log_std)
        return z_sample

    def recon_edge(self, inputs):
        """Run the model."""
        latent_sample = self.call(inputs)
        return recon_edge_helper(latent_sample)


class VGAE_tfp1(tf.keras.Model):
    def __init__(self):
        super(VGAE_tfp1, self).__init__()
        self.conv1 = GraphConvolution(
            16, 1 , activation='relu', kernel_regularizer=l2(5e-4), name='conv1'
        )
        self.conv2 = GraphConvolution(
            16, 1, name='conv2'
        )
        self.dense1 = tkl.Dense(tfpl.MultivariateNormalTriL.params_size(7))
        self.prior = tfd.Independent(tfd.Normal(loc=tf.zeros(7), scale=1),
                        reinterpreted_batch_ndims=1)
        self.dist1 = tfpl.MultivariateNormalTriL(7,
            activity_regularizer=tfpl.KLDivergenceRegularizer(self.prior))


    def call(self, inputs):
        X = tf.cast(tf.convert_to_tensor(inputs[0]), tf.float32)
        G = tf.cast(convert_sparse_matrix_to_sparse_tensor(inputs[1]), tf.float32)
        inputs_tensor = [X, G]
        latent = self.conv1(inputs_tensor)
        latent = self.conv2([latent, G])
        dist_params = self.dense1(latent)
        dist = self.dist1(dist_params)
        return dist

    def recon_edge(self, inputs):
        """Run the model."""
        latent_sample = self.call(inputs)
        return recon_edge_helper(latent_sample)

class VGAE_tfp2(tf.keras.Model):
    def __init__(self):
        super(VGAE_tfp2, self).__init__()
        self.conv1 = GraphConvolution(
            16, 1, activation='relu', kernel_regularizer=l2(5e-4), name='conv1'
        )
        self.conv2 = GraphConvolution(
            7, 1, name='conv2'
        )
        self.conv3 = GraphConvolution(
            7, 1, name='conv2'
        )
        self.prior = tfd.Independent(tfd.Normal(loc=tf.zeros(7), scale=1),
                                     reinterpreted_batch_ndims=1)


    def call(self, inputs):
        X = tf.cast(tf.convert_to_tensor(inputs[0]), tf.float32)
        G = tf.cast(convert_sparse_matrix_to_sparse_tensor(inputs[1]), tf.float32)
        inputs_tensor = [X, G]
        latent = self.conv1(inputs_tensor)
        self.z_mean = self.conv2([latent, G])
        self.z_log_std = self.conv3([latent, G])
        z_sample = tfd.MultivariateNormalDiag(loc=self.z_mean, scale_diag=self.z_log_std)
        # Todo: how to add kl in tfp
        return z_sample

    def recon_edge(self, inputs):
        """Run the model."""
        latent_sample = self.call(inputs)
        return recon_edge_helper(latent_sample)


############################################# GAE ###############################################################

gae = GAE()

optimizer = tf.train.AdamOptimizer()

loss_history = []

for epoch in tqdm(range(NB_EPOCH)):
    with tf.GradientTape() as tape:
        logits = gae.recon_edge(graph)
        loss_value = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
            labels=tf.cast(tf.convert_to_tensor(np.asarray(A.toarray()).reshape(-1)), tf.float32),
            logits=tf.cast(tf.convert_to_tensor(logits), tf.float32),
            pos_weight=pos_weight
        ))

    loss_history.append(loss_value.numpy())
    grads = tape.gradient(loss_value, gae.trainable_variables)
    optimizer.apply_gradients(zip(grads, gae.trainable_variables),
                            global_step=tf.train.get_or_create_global_step())

plt.plot(loss_history)
plt.xlabel('Batch #')
plt.ylabel('Loss [entropy]')
# plt.show()
plt.savefig('gae_loss.png')
plt.clf()
plt.close()
embeddings = gae(graph)

visualzie_embeddding(embeddings, labels, 'gae_cluster.png')

############################################# VGAE ###############################################################

gae = VGAE()

optimizer = tf.train.AdamOptimizer()

loss_history = []

for epoch in tqdm(range(NB_EPOCH)):
    with tf.GradientTape() as tape:
        logits = gae.recon_edge(graph)
        loss_recon = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
            labels=tf.cast(tf.convert_to_tensor(np.asarray(A.toarray()).reshape(-1)), tf.float32),
            logits=tf.cast(tf.convert_to_tensor(logits), tf.float32),
            pos_weight=pos_weight
        ))
        loss_kl = -(0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * gae.z_log_std - tf.square(gae.z_mean) -
                                                                   tf.square(tf.exp(gae.z_log_std)), 1))
        loss_value = loss_recon + loss_kl
    loss_history.append(loss_value.numpy())
    grads = tape.gradient(loss_value, gae.trainable_variables)
    optimizer.apply_gradients(zip(grads, gae.trainable_variables),
                            global_step=tf.train.get_or_create_global_step())

plt.plot(loss_history)
plt.xlabel('Batch #')
plt.ylabel('Loss [entropy]')
plt.savefig('vgae_loss.png')
plt.clf()
plt.close()
embeddings = gae(graph)

visualzie_embeddding(embeddings, labels, 'vgae_cluster.png')

############################################ VGAE_tfp1 ###############################################################

gae = VGAE_tfp1()

optimizer = tf.train.AdamOptimizer()

loss_history = []

for epoch in tqdm(range(NB_EPOCH)):
    with tf.GradientTape() as tape:
        logits = gae.recon_edge(graph)
        loss_value = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
            labels=tf.cast(tf.convert_to_tensor(np.asarray(A.toarray()).reshape(-1)), tf.float32),
            logits=tf.cast(tf.convert_to_tensor(logits), tf.float32),
            pos_weight=pos_weight
        ))

    loss_history.append(loss_value.numpy())
    grads = tape.gradient(loss_value, gae.trainable_variables)
    optimizer.apply_gradients(zip(grads, gae.trainable_variables),
                            global_step=tf.train.get_or_create_global_step())

plt.plot(loss_history)
plt.xlabel('Batch #')
plt.ylabel('Loss [entropy]')
plt.savefig('vgae_tfp1_loss.png')
plt.clf()
plt.close()
embeddings = gae(graph)

visualzie_embeddding(embeddings, labels, 'vgae_tfp1_cluster.png')


############################################ VGAE_tfp2 ###############################################################

gae = VGAE_tfp2()

optimizer = tf.train.AdamOptimizer()

loss_history = []

for epoch in tqdm(range(NB_EPOCH)):
    with tf.GradientTape() as tape:
        logits = gae.recon_edge(graph)
        loss_recon = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
            labels=tf.cast(tf.convert_to_tensor(np.asarray(A.toarray()).reshape(-1)), tf.float32),
            logits=tf.cast(tf.convert_to_tensor(logits), tf.float32),
            pos_weight=pos_weight
        ))
        loss_kl = -(0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * gae.z_log_std - tf.square(gae.z_mean) -
                                                                    tf.square(tf.exp(gae.z_log_std)), 1))
        loss_value = loss_recon + loss_kl

    loss_history.append(loss_value.numpy())
    grads = tape.gradient(loss_value, gae.trainable_variables)
    optimizer.apply_gradients(zip(grads, gae.trainable_variables),
                            global_step=tf.train.get_or_create_global_step())

plt.plot(loss_history)
plt.xlabel('Batch #')
plt.ylabel('Loss [entropy]')
plt.savefig('vgae_tfp2_loss.png')
plt.clf()
plt.close()
embeddings = gae(graph)

visualzie_embeddding(embeddings, labels, 'vgae_tfp2_cluster.png')