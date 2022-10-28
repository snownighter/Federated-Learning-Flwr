import pandas as pd
import numpy as np
# preprocess
from sklearn.feature_extraction import FeatureHasher
from sklearn.compose import ColumnTransformer
from sklearn.utils import shuffle

import tensorflow as tf

sv = "127.0.0.1:8080" # server-address
rs = 3 # num-rounds
model = 'MobileNetV2' # model-name

def load_data(num: int):
    # read: train(client), test, vector_size
    train = pd.read_csv('../data/data-client' + str(num) + '.csv')
    size = pd.read_csv('../data/minmax-train.csv')
    test = pd.read_csv('../data/minmax-test.csv')
    # reprocess
    x_train, y_train = preprocess(shuffle(train), size)
    x_test, y_test = preprocess(shuffle(test), size)
    return (x_train, y_train), (x_test, y_test)

def preprocess(data: pd.DataFrame, size):
    # set columns_name list
    str_name = ['src_ip', 'dst_ip', 'server_port', 'prot']
    num_name = ['p_count', 'b_count', 'max_size', 'min_size', 'abyte_count', 'sbyte_count']
    # 分割資料: string-type feature, label
    X_str, Y_lab = data.loc[:, str_name], data.iloc[:, 10]
    # [label]: One-hot編碼
    Y_lab = pd.get_dummies(Y_lab, prefix_sep='')
    # [feature]: hash-vector transformer
    vector_size = np.zeros(4, dtype=np.int32) # set hash-vector
    for i in range(4): # static vector-size
        vector_size[i] = len(size.iloc[:,i].unique())
    ct = ColumnTransformer([ # 4x hash-vector in ColumnTransformer
        ('src_ip',      FeatureHasher(n_features=vector_size[0], input_type='string'), 0),
        ('dst_ip',      FeatureHasher(n_features=vector_size[1], input_type='string'), 1),
        ('server_port', FeatureHasher(n_features=vector_size[2], input_type='string'), 2),
        ('proto',       FeatureHasher(n_features=vector_size[3], input_type='string'), 3)])
    str_t = X_str.iloc[:,0:4].astype('str') # string-type
    X_str = pd.DataFrame(ct.fit_transform(str_t).toarray()) # One-hot編碼
    return pd.concat([X_str, data.loc[:, num_name]], axis=1), Y_lab # 合併資料

#def modelcompile(mdel: tf.keras.applications):
#    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

def load_mydata():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    return (x_train, y_train), (x_test, y_test)

def new_model(name: str):
    if name == 'MobileNetV2':
        model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
    return model
