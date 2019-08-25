from __future__ import print_function
import tensorflow as tf
tf.enable_eager_execution()
from keras.layers import Input, Dropout, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
import keras.backend as K
from layers.graph import GraphConvolution
from utils import *
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
import time

# Define parameters
DATASET = 'cora'
FILTER = 'localpool'  # 'chebyshev'
MAX_DEGREE = 2  # maximum polynomial degree
SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
NB_EPOCH = 200
PATIENCE = 10  # early stopping patience

# Get data
X, A, y = load_data(dataset=DATASET)
y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits(y)

pos_weight = float(A.shape[0] * A.shape[0] - A.sum()) / A.sum()

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

class GAE(tf.keras.Model):
    def __init__(self):
        super(GAE, self).__init__()
        self.drop1 = Dropout(0.5)
        self.conv1 = GraphConvolution(
            16, 1 , activation='relu', kernel_regularizer=l2(5e-4), name='conv1'
        )
        self.drop2 = Dropout(0.5)
        self.conv2 = GraphConvolution(
            7, 1, name='conv2'
        )


    def call(self, inputs):
        X = tf.cast(tf.convert_to_tensor(inputs[0]), tf.float32)
        G = tf.cast(convert_sparse_matrix_to_sparse_tensor(inputs[1]), tf.float32)
        inputs_tensor = [self.drop1(X), G]
        result = self.conv1(inputs_tensor)
        result = self.conv2([self.drop2(result), G])
        return result

    def recon_edge_logits(self, inputs):
        """Run the model."""
        latent = self.call(inputs)
        result = tf.reshape(tf.matmul(latent, tf.transpose(latent)), [-1])
        return tf.nn.sigmoid(result)


gae = GAE()

optimizer = tf.train.AdamOptimizer()

loss_history = []

for epoch in tqdm(range(400)):
    with tf.GradientTape() as tape:
        logits = gae.recon_edge_logits(graph)
        loss_value = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
            labels=tf.cast(tf.convert_to_tensor(np.asarray(A.toarray()).reshape(-1)), tf.float32),
            logits=tf.cast(tf.convert_to_tensor(logits), tf.float32),
            pos_weight=pos_weight
        ))

    loss_history.append(loss_value.numpy())
    grads = tape.gradient(loss_value, gae.trainable_variables)
    optimizer.apply_gradients(zip(grads, gae.trainable_variables),
                            global_step=tf.train.get_or_create_global_step())

import matplotlib.pyplot as plt

plt.plot(loss_history)
plt.xlabel('Batch #')
plt.ylabel('Loss [entropy]')
plt.show()

embeddings = gae(graph).numpy()

X_embedded = PCA().fit_transform(embeddings)

labels = np.argmax(y, 1)

for l in set(labels):
    plt.scatter(X_embedded[np.argwhere(labels == l), 0], X_embedded[np.argwhere(labels == l), 1], label=l)
plt.legend()
plt.show()