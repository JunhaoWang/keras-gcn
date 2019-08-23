
from __future__ import print_function
import tensorflow as tf
tf.enable_eager_execution()
from keras.layers import Input, Dropout, Dot, Lambda, Activation
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from sklearn.utils import class_weight
from layers.graph import GraphConvolution
from utils import *

import time

# Define parameters
DATASET = 'cora'
FILTER = 'localpool'  # 'chebyshev'
MAX_DEGREE = 2  # maximum polynomial degree
SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
NB_EPOCH = 500
PATIENCE = 100  # early stopping patience

# Get data
X, A, y = load_data(dataset=DATASET)
y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits(A.toarray())

# Normalize X
X /= X.sum(1).reshape(-1, 1)

# if FILTER == 'localpool':
#     """ Local pooling filters (see 'renormalization trick' in Kipf & Welling, arXiv 2016) """
#     print('Using local pooling filters...')
#     A_ = preprocess_adj(A, SYM_NORM)
#     support = 1
#     graph = [X, A_]
#     G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True)]

# elif FILTER == 'chebyshev':
#     """ Chebyshev polynomial basis filters (Defferard et al., NIPS 2016)  """
#     print('Using Chebyshev polynomial basis filters...')
#     L = normalized_laplacian(A, SYM_NORM)
#     L_scaled = rescale_laplacian(L)
#     T_k = chebyshev_polynomial(L_scaled, MAX_DEGREE)
#     support = MAX_DEGREE + 1
#     graph = [X]+T_k
#     G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True) for _ in range(support)]

# else:
#     raise Exception('Invalid filter type.')

pos_weight = float(A.shape[0] * A.shape[0] - A.sum()) / A.sum()

class GAE(tf.keras.Model):
    def __init__(self, X, A, dim, directed):
        super(tf.keras.Model, self).__init__()
        self.X = np.asarray(X)
        self.X_normalized = np.asarray(X / X.sum(1).reshape(-1, 1))
        self.A = A.toarray()
        self.A_processed = preprocess_adj(A, not directed).toarray()
        self.conv1 = GraphConvolution(dim, 1, activation='relu', kernel_regularizer=l2(5e-4))
        self.conv2 = GraphConvolution(dim, 1, activation='relu', kernel_regularizer=l2(5e-4))
    def call(self, inputs):
        h1 = self.conv1([tf.cast(tf.convert_to_tensor(self.X_normalized), tf.double),
                         tf.cast(tf.convert_to_tensor(self.A_processed), tf.double)])
        print(h1)
        h2 = self.conv2([h1, self.A_processed])
        return h2

gae = GAE(X,A,16,False)

gae(2)