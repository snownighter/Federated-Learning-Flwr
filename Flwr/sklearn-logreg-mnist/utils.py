from typing import Tuple, Union, List
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import FeatureHasher
from sklearn.compose import ColumnTransformer
from sklearn.utils import shuffle as pd_shuffle
import openml

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras import optimizers

from sklearn.neural_network import MLPClassifier

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]

def create_model():
    model = MLPClassifier(
        solver = 'adam', # optimizer
        hidden_layer_sizes = (256, 256, 256), # hidden_layer
        activation = 'relu', # for the hidden_layer
        validation_fraction = 0.2, # validation_split
        max_iter = 100, # epochs
        batch_size = 4096, # batch_size
        learning_rate_init = 0.01, # learning_rate
        random_state = 1,
        verbose = False # verbose
    )
    return model

def get_model_parameters(model: MLPClassifier) -> LogRegParams:
    """Returns the paramters of a sklearn LogisticRegression model."""
    params = [
        model.coef_,
        model.intercept_,
    ]
    return params


def set_model_params(
    model: MLPClassifier, params: LogRegParams
) -> MLPClassifier:
    """Sets the parameters of a sklean LogisticRegression model."""
    model.coef_ = params[0]
    model.intercept_ = params[1]
    return model


def set_initial_params(model: MLPClassifier):
    """Sets initial parameters as zeros Required since model params are
    uninitialized until model.fit is called.

    But server asks for initial parameters from clients at launch. Refer
    to sklearn.linear_model.LogisticRegression documentation for more
    information.
    """
    n_classes = 25  # MNIST has 10 classes
    n_features = 3073  # Number of features in dataset
    model.classes_ = np.array([i for i in range(25)])

    model.coef_ = np.zeros((n_classes, n_features))
    model.intercept_ = np.zeros((n_classes,))

def load_data(num):
    # 載入
    train = pd.read_csv('../data/data-client' + str(num) + '.csv')
    size = pd.read_csv('../data/minmax-train.csv')
    test = pd.read_csv('../data/minmax-test.csv')
    x_train, y_train = preprocess(pd_shuffle(train), size)
    x_test, y_test = preprocess(pd_shuffle(test), size)

    return (x_train, y_train), (x_test, y_test)

def preprocess(data, size):
    str_name = ['src_ip', 'dst_ip', 'server_port', 'prot']
    num_name = ['p_count', 'b_count', 'max_size', 'min_size', 'abyte_count', 'sbyte_count']

    X_str, Y_lab = data.loc[:, str_name], data.iloc[:, 10] # 分割資料
    Y_lab = pd.get_dummies(Y_lab, prefix_sep='') # One-hot編碼

    vector_size = np.zeros(4, dtype=np.int32) # 設置hash空間
    for i in range(4):
        vector_size[i] = len(size.iloc[:,i].unique()) # 限定train大小 (使train和test相同大小)

    ct = ColumnTransformer([
        ('src_ip',      FeatureHasher(n_features=vector_size[0], input_type='string'), 0),
        ('dst_ip',      FeatureHasher(n_features=vector_size[1], input_type='string'), 1),
        ('server_port', FeatureHasher(n_features=vector_size[2], input_type='string'), 2),
        ('proto',       FeatureHasher(n_features=vector_size[3], input_type='string'), 3)])
    str_t = X_str.iloc[:,0:4].astype('str') 
    X_str = pd.DataFrame(ct.fit_transform(str_t).toarray()) # hash 之 One-hot編碼
    
    return pd.concat([X_str, data.loc[:, num_name]], axis=1), Y_lab # 合併資料

def load_mnist() -> Dataset:
    """Loads the MNIST dataset using OpenML.

    OpenML dataset link: https://www.openml.org/d/554
    """
    mnist_openml = openml.datasets.get_dataset(554)
    Xy, _, _, _ = mnist_openml.get_data(dataset_format="array")
    X = Xy[:, :-1]  # the last column contains labels
    y = Xy[:, -1]
    # First 60000 samples consist of the train set
    x_train, y_train = X[:60000], y[:60000]
    x_test, y_test = X[60000:], y[60000:]
    return (x_train, y_train), (x_test, y_test)


def shuffle(X: np.ndarray, y: np.ndarray) -> XY:
    """Shuffle X and y."""
    rng = np.random.default_rng()
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    """Split X and y into a number of partitions."""
    return list(
        zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions))
    )
