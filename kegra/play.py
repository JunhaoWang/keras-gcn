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
import time
from sklearn.metrics.cluster import normalized_mutual_info_score
import networkx as nx
from functools import reduce
import scipy.sparse as sp

np.random.seed(0)

# Define parameters
DATASET = 'cora'
FILTER = 'localpool'  # 'chebyshev'
MAX_DEGREE = 2  # maximum polynomial degree
SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
NB_EPOCH = 4000
PATIENCE = 10  # early stopping patience
BETA_VAE = 1
VISAUL_FREQ = 100
NUM_SAMPLE = 5

# Get data
# ############################################## COMMUNITY ##################################################
path_to_dataset = 'data_comm/medicine.npz'
# Load the data
G = load_dataset_comm(path_to_dataset)
A, X, F = G['A'], G['X'], G['F']
labels = np.argmax(F.toarray(), 1)
# X = X.toarray()[:500,:5]
# A = sp.csr_matrix(A[:500,:][:,:500].toarray())
# labels = labels[:500]

# ############################################## CORA ##################################################
# X, A, y = load_data(dataset=DATASET)
# labels = np.argmax(y, 1)

# ############################################## SBM ##################################################
#
# ## diff 3
# sizes = [200, 200, 200, 200]
# self_probs = [0.01, 0.03, 0.05, 0.07]
#
# scale_factor = .1
# probs = np.diag(self_probs)
# probs = np.array(self_probs).reshape(-1, 1) * scale_factor + probs
# g = nx.stochastic_block_model(sizes, probs, directed = True, seed=0)
#
# X = np.eye(np.sum(sizes))
# A = nx.to_scipy_sparse_matrix(g, dtype=np.float32)
# labels = np.array(
#     reduce(lambda  x,y: x+y, [[ind] * i for ind, i in enumerate(sizes)])
# )

#############################################################################################

pos_weight = float(A.shape[0] * A.shape[0] - A.sum()) / A.sum()
norm = A.shape[0] * A.shape[0] / float((A.shape[0] * A.shape[0] - A.sum()) * 2)
num_nodes = A.shape[0]


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

def make_gaussian_mixture_prior(latent_size, mixture_components):
  """Creates the mixture of Gaussians prior distribution.
  Args:
    latent_size: The dimensionality of the latent representation.
    mixture_components: Number of elements of the mixture.
  Returns:
    random_prior: A `tfd.Distribution` instance representing the distribution
      over encodings in the absence of any evidence.
  """
  if mixture_components == 1:
    # See the module docstring for why we don't learn the parameters here.
    return tfd.MultivariateNormalDiag(
        loc=tf.zeros([latent_size]),
        scale_identity_multiplier=1.0)

  loc = tf.compat.v1.get_variable(
      name="loc", shape=[mixture_components, latent_size])
  raw_scale_diag = tf.compat.v1.get_variable(
      name="raw_scale_diag", shape=[mixture_components, latent_size])
  mixture_logits = tf.compat.v1.get_variable(
      name="mixture_logits", shape=[mixture_components])

  components = tfp.distributions.MultivariateNormalDiag(
      loc=loc,
      scale_diag=tf.nn.softplus(raw_scale_diag))

  return tfp.distributions.MixtureSameFamily(
      components_distribution=components,
      mixture_distribution=tfd.Categorical(logits=mixture_logits))


def make_gaussian_mixture_prior_lowrank(latent_size, mixture_components, perturb_rank):
  """Creates the mixture of Gaussians prior distribution.
  Args:
    latent_size: The dimensionality of the latent representation.
    mixture_components: Number of elements of the mixture.
  Returns:
    random_prior: A `tfd.Distribution` instance representing the distribution
      over encodings in the absence of any evidence.
  """
  if mixture_components == 1:
    # See the module docstring for why we don't learn the parameters here.
    return tfd.MultivariateNormalDiag(
        loc=tf.zeros([latent_size]),
        scale_identity_multiplier=1.0)

  loc = tf.compat.v1.get_variable(
      name="loc", shape=[mixture_components, latent_size])
  raw_scale_diag = tf.compat.v1.get_variable(
      name="raw_scale_diag", shape=[mixture_components, latent_size])
  mixture_logits = tf.compat.v1.get_variable(
      name="mixture_logits", shape=[mixture_components])
  perturb_U = tf.compat.v1.get_variable(
      name="perturb_U", shape=[mixture_components, latent_size, perturb_rank])
  perturb_m = tf.compat.v1.get_variable(
      name="perturb_m", shape=[mixture_components, perturb_rank])

  components = tfp.distributions.MultivariateNormalDiagPlusLowRank(
      loc=loc,
      scale_diag=tf.nn.softplus(raw_scale_diag),
      scale_perturb_factor=perturb_U,
      scale_perturb_diag=perturb_m
  )

  return tfp.distributions.MixtureSameFamily(
      components_distribution=components,
      mixture_distribution=tfd.Categorical(logits=mixture_logits))

class dpmeans:

    def __init__(self, X):
        # Initialize parameters for DP means

        self.K = 1
        self.K_init = 4
        self.d = X.shape[1]
        self.z = np.mod(np.random.permutation(X.shape[0]), self.K) + 1
        self.mu = np.random.standard_normal((self.K, self.d))
        self.sigma = 1
        self.nk = np.zeros(self.K)
        self.pik = np.ones(self.K) / self.K

        # init mu
        self.mu = np.array([np.mean(X, 0)])

        # init lambda
        self.Lambda = self.kpp_init(X, self.K_init)

        self.max_iter = 100
        self.obj = np.zeros(self.max_iter)
        self.em_time = np.zeros(self.max_iter)

    def kpp_init(self, X, k):
        # k++ init
        # lambda is max distance to k++ means

        [n, d] = np.shape(X)
        mu = np.zeros((k, d))
        dist = np.inf * np.ones(n)

        mu[0, :] = X[int(np.ceil(np.random.rand() * n - 1)), :]
        for i in range(1, k):
            D = X - np.tile(mu[i - 1, :], (n, 1))
            dist = np.minimum(dist, np.sum(D * D, 1))
            idx = np.where(np.random.rand() < np.cumsum(dist / float(sum(dist))))
            mu[i, :] = X[idx[0][0], :]
            Lambda = np.max(dist)

        return Lambda

    def fit(self, X):

        obj_tol = 1e-3
        max_iter = self.max_iter
        [n, d] = np.shape(X)

        obj = np.zeros(max_iter)
        em_time = np.zeros(max_iter)
        print('running dpmeans...')

        for iter in range(max_iter):
            tic = time.time()
            dist = np.zeros((n, self.K))

            # assignment step
            for kk in range(self.K):
                Xm = X - np.tile(self.mu[kk, :], (n, 1))
                dist[:, kk] = np.sum(Xm * Xm, 1)

            # update labels
            dmin = np.min(dist, 1)
            self.z = np.argmin(dist, 1)
            idx = np.where(dmin > self.Lambda)

            if (np.size(idx) > 0):
                self.K = self.K + 1
                self.z[idx[0]] = self.K - 1  # cluster labels in [0,...,K-1]
                self.mu = np.vstack([self.mu, np.mean(X[idx[0], :], 0)])
                Xm = X - np.tile(self.mu[self.K - 1, :], (n, 1))
                dist = np.hstack([dist, np.array([np.sum(Xm * Xm, 1)]).T])

            # update step
            self.nk = np.zeros(self.K)
            for kk in range(self.K):
                self.nk[kk] = self.z.tolist().count(kk)
                idx = np.where(self.z == kk)
                self.mu[kk, :] = np.mean(X[idx[0], :], 0)

            self.pik = self.nk / float(np.sum(self.nk))

            # compute objective
            for kk in range(self.K):
                idx = np.where(self.z == kk)
                obj[iter] = obj[iter] + np.sum(dist[idx[0], kk], 0)
            obj[iter] = obj[iter] + self.Lambda * self.K

            # check convergence
            if (iter > 0 and np.abs(obj[iter] - obj[iter - 1]) < obj_tol * obj[iter]):
                print('converged in %d iterations\n' % iter)
                break
            em_time[iter] = time.time() - tic
        # end for
        self.obj = obj
        self.em_time = em_time
        return self.z, obj, em_time

    def compute_nmi(self, z1, z2):
        # compute normalized mutual information

        n = np.size(z1)
        k1 = np.size(np.unique(z1))
        k2 = np.size(np.unique(z2))

        nk1 = np.zeros((k1, 1))
        nk2 = np.zeros((k2, 1))

        for kk in range(k1):
            nk1[kk] = np.sum(z1 == kk)
        for kk in range(k2):
            nk2[kk] = np.sum(z2 == kk)

        pk1 = nk1 / float(np.sum(nk1))
        pk2 = nk2 / float(np.sum(nk2))

        nk12 = np.zeros((k1, k2))
        for ii in range(k1):
            for jj in range(k2):
                nk12[ii, jj] = np.sum((z1 == ii) * (z2 == jj))
        pk12 = nk12 / float(n)

        Hx = -np.sum(pk1 * np.log(pk1 + np.finfo(float).eps))
        Hy = -np.sum(pk2 * np.log(pk2 + np.finfo(float).eps))

        Hxy = -np.sum(pk12 * np.log(pk12 + np.finfo(float).eps))

        MI = Hx + Hy - Hxy;
        nmi = MI / float(0.5 * (Hx + Hy))

        return nmi

    def generate_plots(self, X):

        plt.close('all')
        plt.figure(0)
        for kk in range(self.K):
            # idx = np.where(self.z == kk)
            plt.scatter(X[self.z == kk, 0], X[self.z == kk, 1], \
                        s=100, marker='o', label=str(kk))
        # end for
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()
        plt.title('DP-means clusters')
        plt.grid(True)
        plt.show()

        plt.figure(1)
        plt.plot(self.obj)
        plt.title('DP-means objective function')
        plt.xlabel('iterations')
        plt.ylabel('penalized l2 squared distance')
        plt.grid(True)
        plt.show()

        plt.figure(2)
        plt.plot(self.em_time)
        plt.title('DP-means time per iteration')
        plt.xlabel('iterations')
        plt.ylabel('time, sec')
        plt.grid(True)
        plt.show()

    def display_params(self):

        print('K = %d' % self.K)
        print('d = %d' % self.d)
        print('Labels:')
        print(self.z)
        print('Means:')
        print(self.mu)
        print('Sigma:')
        print(self.sigma)
        print('Counts:')
        print(self.nk)
        print('Proportions:')
        print(self.pik)
        print('Lambda: %.2f' % self.Lambda)


def visualize_embedding_helper(X_embedded, labels, name):
    for l in set(labels):
        plt.scatter(X_embedded[np.argwhere(labels == l), 0], X_embedded[np.argwhere(labels == l), 1], label=l)
    plt.legend()
    plt.savefig(name)
    plt.clf()
    plt.close()

def visualzie_embeddding(embeddings, labels, name):
    if tf.is_tensor(embeddings):
        embeddings = embeddings.numpy()
    else:
        embeddings = embeddings.mean().numpy()
    X_embedded = PCA().fit_transform(embeddings)
    visualize_embedding_helper(X_embedded, labels, name)

    dp = dpmeans(embeddings)
    inferred_labels, obj, em_time = dp.fit(embeddings)
    # nmi = dp.compute_nmi(inferred_labels, labels)
    print('dirichlet process cluster NMI: {}'.format(normalized_mutual_info_score(labels, inferred_labels)))
    visualize_embedding_helper(X_embedded, inferred_labels, 'dirichlet_inferred_' + name)

class GAE(tf.keras.Model):
    def __init__(self):
        super(GAE, self).__init__()
        self.conv1 = GraphConvolution(
            16, 1 , activation='relu', kernel_regularizer=l2(5e-4)
        )
        self.conv2 = GraphConvolution(
            7, 1
        )

    def call(self, inputs):
        X = tf.cast(convert_sparse_matrix_to_sparse_tensor(inputs[0]), tf.float32)
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
            16, 1 , activation='relu', kernel_regularizer=l2(5e-4)
        )
        self.conv2 = GraphConvolution(
            7, 1
        )
        self.conv3 = GraphConvolution(
            7, 1
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
            16, 1 , activation='relu', kernel_regularizer=l2(5e-4)
        )
        self.conv2 = GraphConvolution(
            16, 1
        )
        self.dense1 = tkl.Dense(tfpl.MultivariateNormalTriL.params_size(7))
        self.prior = tfd.Independent(tfd.Normal(loc=tf.zeros(7), scale=1),
                        reinterpreted_batch_ndims=1)
        self.dist1 = tfpl.MultivariateNormalTriL(7,
            activity_regularizer=tfpl.KLDivergenceRegularizer(self.prior, weight=BETA_VAE))


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
            16, 1, activation='relu', kernel_regularizer=l2(5e-4)
        )
        self.conv2 = GraphConvolution(
            7, 1
        )
        self.conv3 = GraphConvolution(
            7, 1
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


class MDGAE(tf.keras.Model):
    # Todo: try temperature (work kinda), tf contrib loss (trying);
    #  sampling instead of marginalizing (not working), kian's loss (not yet)
    def __init__(self, latent_dim = 7, num_component = 7):
        super(MDGAE, self).__init__()
        self.num_component = num_component
        self.latent_dim = latent_dim
        self.conv1 = GraphConvolution(
            self.latent_dim * 2, 1 , activation='relu', kernel_regularizer=l2(5e-4)
        )
        self.conv2 = GraphConvolution(
            self.num_component, 1, activation='softmax'
        )
        self.conv3 = GraphConvolution(
            self.num_component, 1, activation='softplus'
        )
        self.conv4 = GraphConvolution(
            self.latent_dim * self.num_component, 1
        )
        # self.alpha_temperature = tf.get_variable("alpha_temperature", [1],
        #                         dtype=tf.float32,initializer=tf.zeros_initializer)
        # self.base_temperature = tf.get_variable("base_temperature", [1],
        #                         dtype=tf.float32,initializer=tf.zeros_initializer)

    def call(self, inputs):
        X = tf.cast(tf.convert_to_tensor(inputs[0]), tf.float32)
        G = tf.cast(convert_sparse_matrix_to_sparse_tensor(inputs[1]), tf.float32)
        inputs_tensor = [X, G]
        latent = self.conv1(inputs_tensor)
        # self.alphas = tf.nn.softmax(
        #     tf.clip_by_value(self.base_temperature, 0.1, 1.0/self.num_component) + \
        #         tf.pow(self.conv2([latent, G]), 1 + tf.clip_by_value(self.alpha_temperature, .5, 1))
        # )
        self.alphas = self.conv2([latent, G])
        self.z_log_std = self.conv3([latent, G])
        z_mean_mix = self.conv4([latent, G])
        batch_size = z_mean_mix.shape.as_list()[0]
        self.z_mean_mix_reshape = tf.reshape(z_mean_mix, [batch_size, self.num_component, self.latent_dim])
        z_std = tf.exp(self.z_log_std)
        z_std_reshape = tf.stack([z_std] * self.num_component, axis=-1)
        z_sample_all = self.z_mean_mix_reshape + tf.random_normal(self.z_mean_mix_reshape.shape.as_list()) * z_std_reshape
        z_sample_marginalized = tf.matmul(z_sample_all, tf.expand_dims(
            self.alphas, -1
        ))
        return tf.squeeze(z_sample_marginalized)

    def recon_edge(self, inputs):
        """Run the model."""
        latent_sample = self.call(inputs)
        return recon_edge_helper(latent_sample)


class MDGAE_tfp1(tf.keras.Model):
    def __init__(self, latent_dim = 7, num_component = 7):
        super(MDGAE_tfp1, self).__init__()
        self.num_component = num_component
        self.latent_dim = latent_dim
        self.conv1 = GraphConvolution(
            self.latent_dim * 2, 1 , activation='relu', kernel_regularizer=l2(5e-4)
        )
        self.conv2 = GraphConvolution(
            self.latent_dim * 2, 1, activation='relu'
        )
        self.dense1 = tf.keras.layers.Dense(2 * latent_dim)
        self.prior = make_gaussian_mixture_prior(self.latent_dim, self.num_component)

    def call(self, inputs):
        X = tf.cast(tf.convert_to_tensor(inputs[0]), tf.float32)
        G = tf.cast(convert_sparse_matrix_to_sparse_tensor(inputs[1]), tf.float32)
        inputs_tensor = [X, G]
        latent = self.conv1(inputs_tensor)
        latent = self.conv2([latent, G])
        latent = self.dense1(latent)
        return tfd.MultivariateNormalDiag(
            loc=latent[..., :self.latent_dim],
            scale_diag=tf.nn.softplus(latent[..., self.latent_dim:] +
                                      tfd.softplus_inverse(1.0)))

    def recon_edge(self, inputs):
        """Run the model."""
        latent_sample = self.call(inputs)
        return recon_edge_helper(latent_sample)

class MDGAE_tfp2(tf.keras.Model):
    def __init__(self, latent_dim = 7, num_component = 7, perturb_rank = 2):
        super(MDGAE_tfp2, self).__init__()
        self.num_component = num_component
        self.latent_dim = latent_dim
        self.perturb_rank = perturb_rank
        self.conv1 = GraphConvolution(
            self.latent_dim * 2, 1 , activation='relu', kernel_regularizer=l2(5e-4)
        )
        self.conv2 = GraphConvolution(
            self.latent_dim * 2, 1, activation='relu'
        )
        self.dense1 = tf.keras.layers.Dense(2 * self.latent_dim, activation='tanh')
        self.dense2 = tf.keras.layers.Dense((self.latent_dim + 1)* self.perturb_rank, activation='tanh')
        self.prior = make_gaussian_mixture_prior_lowrank(self.latent_dim, self.num_component, self.perturb_rank)

    def call(self, inputs):
        X = tf.cast(tf.convert_to_tensor(inputs[0]), tf.float32)
        G = tf.cast(convert_sparse_matrix_to_sparse_tensor(inputs[1]), tf.float32)
        batch_size = X.shape.as_list()[0]
        inputs_tensor = [X, G]
        latent = self.conv1(inputs_tensor)
        latent = self.conv2([latent, G])
        param_dist = self.dense1(latent)
        param_perturb = self.dense2(latent)
        perturn_U = param_perturb[..., :self.latent_dim * self.perturb_rank]
        perturn_U_reshape = tf.reshape(perturn_U, [batch_size, self.latent_dim, self.perturb_rank])
        perturb_m = param_perturb[..., self.latent_dim * self.perturb_rank:]
        return tfd.MultivariateNormalDiagPlusLowRank(
            loc=param_dist[..., :self.latent_dim],
            scale_diag=tf.nn.softplus(param_dist[..., self.latent_dim:] +
                                      tfd.softplus_inverse(1.0)),
            scale_perturb_factor=perturn_U_reshape,
            scale_perturb_diag=perturb_m
        )

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
        latent_sample = gae(graph)
        # logits = recon_edge_helper(latent_sample)
        # loss_value = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
        #     labels=tf.cast(tf.convert_to_tensor(np.asarray(A.toarray()).reshape(-1)), tf.float32),
        #     logits=tf.cast(tf.convert_to_tensor(logits), tf.float32),
        #     pos_weight=pos_weight
        # ))
        reconstr_loss_stochastic = berpo_loss(latent_sample, A, stochastic=True, batch_size=10000)
        reg_loss = tf.losses.get_regularization_loss()
        loss_value = reconstr_loss_stochastic + reg_loss
    if epoch % VISAUL_FREQ == 1:
        visualzie_embeddding(latent_sample, labels, 'streaming_gae.png')

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

# ############################################# VGAE ###############################################################
#
# gae = VGAE()
#
# optimizer = tf.train.AdamOptimizer()
#
# loss_history = []
#
# for epoch in tqdm(range(NB_EPOCH)):
#     with tf.GradientTape() as tape:
#         latent_sample = gae(graph)
#         logits = recon_edge_helper(latent_sample)
#         loss_recon = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
#             labels=tf.cast(tf.convert_to_tensor(np.asarray(A.toarray()).reshape(-1)), tf.float32),
#             logits=tf.cast(tf.convert_to_tensor(logits), tf.float32),
#             pos_weight=pos_weight
#         ))
#         loss_kl = -(0.5 / num_nodes) * BETA_VAE * tf.reduce_mean(tf.reduce_sum(1 + 2 * gae.z_log_std - tf.square(gae.z_mean) -
#                                                                    tf.square(tf.exp(gae.z_log_std)), 1))
#         loss_value = loss_recon + loss_kl
#     if epoch % VISAUL_FREQ == 1:
#         visualzie_embeddding(latent_sample, labels, 'streaming_vgae.png')
#     loss_history.append(loss_value.numpy())
#     grads = tape.gradient(loss_value, gae.trainable_variables)
#     optimizer.apply_gradients(zip(grads, gae.trainable_variables),
#                             global_step=tf.train.get_or_create_global_step())
#
# plt.plot(loss_history)
# plt.xlabel('Batch #')
# plt.ylabel('Loss [entropy]')
# plt.savefig('vgae_loss.png')
# plt.clf()
# plt.close()
# embeddings = gae(graph)
#
# visualzie_embeddding(embeddings, labels, 'vgae_cluster.png')
#
# ############################################ VGAE_tfp1 ###############################################################
#
# gae = VGAE_tfp1()
#
# optimizer = tf.train.AdamOptimizer()
#
# loss_history = []
#
# for epoch in tqdm(range(NB_EPOCH)):
#     with tf.GradientTape() as tape:
#         latent_sample = gae(graph)
#         logits = recon_edge_helper(latent_sample)
#         loss_value = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
#             labels=tf.cast(tf.convert_to_tensor(np.asarray(A.toarray()).reshape(-1)), tf.float32),
#             logits=tf.cast(tf.convert_to_tensor(logits), tf.float32),
#             pos_weight=pos_weight
#         ))
#     if epoch % VISAUL_FREQ == 1:
#         visualzie_embeddding(latent_sample, labels, 'streaming_vgae_tfp1.png')
#     loss_history.append(loss_value.numpy())
#     grads = tape.gradient(loss_value, gae.trainable_variables)
#     optimizer.apply_gradients(zip(grads, gae.trainable_variables),
#                             global_step=tf.train.get_or_create_global_step())
#
# plt.plot(loss_history)
# plt.xlabel('Batch #')
# plt.ylabel('Loss [entropy]')
# plt.savefig('vgae_tfp1_loss.png')
# plt.clf()
# plt.close()
# embeddings = gae(graph)
#
# visualzie_embeddding(embeddings, labels, 'vgae_tfp1_cluster.png')
#
#
# ############################################ VGAE_tfp2 ###############################################################
#
# gae = VGAE_tfp2()
#
# optimizer = tf.train.AdamOptimizer()
#
# loss_history = []
#
# for epoch in tqdm(range(NB_EPOCH)):
#     with tf.GradientTape() as tape:
#         latent_sample = gae(graph)
#         logits = recon_edge_helper(latent_sample)
#         loss_recon = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
#             labels=tf.cast(tf.convert_to_tensor(np.asarray(A.toarray()).reshape(-1)), tf.float32),
#             logits=tf.cast(tf.convert_to_tensor(logits), tf.float32),
#             pos_weight=pos_weight
#         ))
#         loss_kl = -(0.5 / num_nodes) * BETA_VAE * tf.reduce_mean(tf.reduce_sum(1 + 2 * gae.z_log_std - tf.square(gae.z_mean) -
#                                                                     tf.square(tf.exp(gae.z_log_std)), 1))
#         loss_value = loss_recon + loss_kl
#     if epoch % VISAUL_FREQ == 1:
#         visualzie_embeddding(latent_sample, labels, 'streaming_vgae_tfp2.png')
#     loss_history.append(loss_value.numpy())
#     grads = tape.gradient(loss_value, gae.trainable_variables)
#     optimizer.apply_gradients(zip(grads, gae.trainable_variables),
#                             global_step=tf.train.get_or_create_global_step())
#
# plt.plot(loss_history)
# plt.xlabel('Batch #')
# plt.ylabel('Loss [entropy]')
# plt.savefig('vgae_tfp2_loss.png')
# plt.clf()
# plt.close()
# embeddings = gae(graph)
#
# visualzie_embeddding(embeddings, labels, 'vgae_tfp2_cluster.png')
#
# ############################################ MDGAE ###############################################################
#
# gae = MDGAE()
#
# optimizer = tf.train.AdamOptimizer()
#
# loss_history = []
#
# for epoch in tqdm(range(NB_EPOCH)):
#     with tf.GradientTape() as tape:
#         latent_sample = gae(graph)
#         logits = recon_edge_helper(latent_sample)
#
#         loss_value = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
#             labels=tf.cast(tf.convert_to_tensor(np.asarray(A.toarray()).reshape(-1)), tf.float32),
#             logits=tf.cast(tf.convert_to_tensor(logits), tf.float32),
#             pos_weight=pos_weight
#         ))
#
#         # loss_recon = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
#         #     labels=tf.cast(tf.convert_to_tensor(np.asarray(A.toarray()).reshape(-1)), tf.float32),
#         #     logits=tf.cast(tf.convert_to_tensor(logits), tf.float32),
#         #     pos_weight=pos_weight
#         # ))
#         #
#         # num_sample = np.min((200, num_nodes))
#         # rand_idx_cluster_loss = np.random.choice(list(range(num_nodes)), num_sample, replace=False)
#         #
#         # cur_inferred_labels = tf.argmax(gae.alphas, axis=1)
#         # cur_inferred_labels_sample = tf.gather(cur_inferred_labels, rand_idx_cluster_loss)
#         # latent_sample_sample = tf.gather(latent_sample, rand_idx_cluster_loss)
#
#         # loss_cluster = tf.contrib.losses.metric_learning.triplet_semihard_loss(
#         #     cur_inferred_labels_sample,
#         #     latent_sample_sample,
#         #     margin=3.0
#         # )
#
#         # loss_value = loss_recon + epoch/100000 * loss_cluster
#         #
#         # if epoch % VISAUL_FREQ == 1:
#         #     X_embedded = PCA().fit_transform(latent_sample.numpy())
#         #     visualize_embedding_helper(X_embedded, cur_inferred_labels.numpy(), 'streaming_mdgae_latent_inferrred_cluster.png')
#         #     print(normalized_mutual_info_score(labels, cur_inferred_labels.numpy()))
#
#         if epoch % VISAUL_FREQ == 1:
#             visualzie_embeddding(latent_sample, labels, 'streaming_mdgae.png')
#
#     loss_history.append(loss_value.numpy())
#     grads = tape.gradient(loss_value, gae.trainable_variables)
#     optimizer.apply_gradients(zip(grads, gae.trainable_variables),
#                             global_step=tf.train.get_or_create_global_step())
#
# embeddings = np.nan_to_num(gae(graph).numpy())
# latent_inferred_labels = np.argmax(gae.alphas.numpy(), 1)
# X_embedded = PCA().fit_transform(embeddings)
# visualize_embedding_helper(X_embedded, latent_inferred_labels, 'mdgae_latent_inferrred_cluster.png')
#
# print('MDGAE NMI latent: {}'.format(normalized_mutual_info_score(labels, latent_inferred_labels)))
# plt.plot(loss_history)
# plt.xlabel('Batch #')
# plt.ylabel('Loss [entropy]')
# plt.savefig('mdgae_loss.png')
# plt.clf()
# plt.close()
# embeddings = gae(graph)
#
# visualzie_embeddding(embeddings, labels, 'mdgae_cluster.png')
#
# ############################################ MDGAE_tfp1 ###############################################################
#
# gae = MDGAE_tfp1()
#
# optimizer = tf.train.AdamOptimizer()
#
# loss_history = []
#
# for epoch in tqdm(range(NB_EPOCH)):
#     with tf.GradientTape() as tape:
#         approx_posterior = gae(graph)
#         logits = recon_edge_helper(approx_posterior)
#         loss_recon = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
#             labels=tf.cast(tf.convert_to_tensor(np.asarray(A.toarray()).reshape(-1)), tf.float32),
#             logits=tf.cast(tf.convert_to_tensor(logits), tf.float32),
#             pos_weight=pos_weight
#         ))
#
#         approx_posterior_sample = approx_posterior.sample(NUM_SAMPLE)
#         loss_kl = (0.5 / num_nodes) * BETA_VAE * tf.reduce_mean(
#             approx_posterior.log_prob(approx_posterior_sample) - gae.prior.log_prob(approx_posterior_sample)
#         )
#         loss_value = loss_recon + loss_kl
#
#     if epoch % VISAUL_FREQ == 1:
#         visualzie_embeddding(approx_posterior, labels, 'streaming_mdgae_tfp1.png')
#     loss_history.append(loss_value.numpy())
#     grads = tape.gradient(loss_value, gae.trainable_variables)
#     optimizer.apply_gradients(zip(grads, gae.trainable_variables),
#                             global_step=tf.train.get_or_create_global_step())
#
# plt.plot(loss_history)
# plt.xlabel('Batch #')
# plt.ylabel('Loss [entropy]')
# plt.savefig('mdgae_tfp1_loss.png')
# plt.clf()
# plt.close()
# embeddings = gae(graph)
#
# visualzie_embeddding(embeddings, labels, 'mdgae_tfp1_cluster.png')
#
# ########################################### MDGAE_tfp2 ###############################################################
#
# gae = MDGAE_tfp2()
#
# optimizer = tf.train.AdamOptimizer()
#
# loss_history = []
#
# for epoch in tqdm(range(NB_EPOCH)):
#     with tf.GradientTape() as tape:
#         approx_posterior = gae(graph)
#         logits = recon_edge_helper(approx_posterior)
#         loss_recon = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
#             labels=tf.cast(tf.convert_to_tensor(np.asarray(A.toarray()).reshape(-1)), tf.float32),
#             logits=tf.cast(tf.convert_to_tensor(logits), tf.float32),
#             pos_weight=pos_weight
#         ))
#
#         approx_posterior_sample = approx_posterior.sample(NUM_SAMPLE)
#         loss_kl = (0.5 / num_nodes) * BETA_VAE * tf.reduce_mean(
#             approx_posterior.log_prob(approx_posterior_sample) - gae.prior.log_prob(approx_posterior_sample)
#         )
#         loss_value = loss_recon + loss_kl
#
#     if epoch % VISAUL_FREQ == 1:
#         visualzie_embeddding(approx_posterior, labels, 'streaming_mdgae_tfp2.png')
#     loss_history.append(loss_value.numpy())
#     grads = tape.gradient(loss_value, gae.trainable_variables)
#     optimizer.apply_gradients(zip(grads, gae.trainable_variables),
#                             global_step=tf.train.get_or_create_global_step())
#
# plt.plot(loss_history)
# plt.xlabel('Batch #')
# plt.ylabel('Loss [entropy]')
# plt.savefig('mdgae_tfp2_loss.png')
# plt.clf()
# plt.close()
# embeddings = gae(graph)
#
# visualzie_embeddding(embeddings, labels, 'mdgae_tfp2_cluster.png')