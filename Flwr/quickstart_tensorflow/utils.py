import pandas as pd
import numpy as np
# preprocess
from sklearn.feature_extraction import FeatureHasher
from sklearn.compose import ColumnTransformer
from sklearn.utils import shuffle

import tensorflow as tf
# model
from tensorflow.python.keras import models, layers
from tensorflow.python.keras import optimizers
# img
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# result
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
# confusion matrix for global model
from sklearn.metrics import confusion_matrix, precision_score, f1_score, recall_score
import seaborn as sns

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

drimg = False # draw-img
drmat = True
wres, gres = True, True # write result
relab = False # unbalance

# The number of ...
N  = 5 # client
GR = 10 # global round
LR = 0.001 # learning rate

class server_fit:
    def __init__(self, round=10, lr=0.001) -> None:
        self.sv = "127.0.0.1:8080" # server-address
        self.rs = round # num-rounds
        self.model = 'Sequential' # model-name
        self.adam = optimizers.adam_v2.Adam(learning_rate=lr) # optimizer
sp = server_fit(GR, LR)

class model_fit:
    def __init__(self, epoch=50) -> None:
        self.epochs = epoch #100
        self.batch_size = 512 #4096
mp, epochlis = [], [50,50,50,50,50] #[1,1,1,1,1] 
for i in range(N):
    mp.append(model_fit(epochlis[i]))

def load_data(num: int):
    # read: train(client), test, vector_size
    train = pd.read_csv('../data/data-client' + str(num) + '.csv')
    size = pd.read_csv('../data/minmax-train.csv')
    test = pd.read_csv('../data/minmax-test.csv')
    # remove
    if relab:
        remove = ['Facebook', 'GMail', 'GoogleDrive', 'Instagram']
        train.iloc[train.iloc[:,10]==remove[num-1], 0:10] = 0
    # reprocess
    x_train, y_train = preprocess(shuffle(train), size)
    x_test, y_test = preprocess(shuffle(test), size)
    return (x_train, y_train), (x_test, y_test)

def preprocess(data: pd.DataFrame, size):
    # set columns_name list
    str_name = ['src_ip', 'dst_ip', 'server_port', 'prot']
    num_name = ['p_count', 'b_count', 'max_size', 'min_size', 'abyte_count', 'sbyte_count']
    # 分割資料: string-type feature, label
    X_str, Y_orlab = data.loc[:, str_name], data.iloc[:, 10]
    # [label]: One-hot編碼
    Y_lab = pd.get_dummies(Y_orlab, prefix_sep='')
    # reshape [label]
    label = list(sorted(set(size.iloc[:, 10])))
    lab = list(sorted(set(Y_orlab)))
    li = []
    for i in label:
        if i not in lab: li.append(label.index(i)) #print('ex', i)
    for i in li: Y_lab.insert(i, label[i], 0)
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

def new_model(name: str, input_size: int):
    if   name == 'MobileNetV2':
        model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
    elif name == 'Sequential':
        It = tf.random_normal_initializer()
        model = models.Sequential()
        model.add(layers.Dense(256, input_dim=input_size, activation='relu', kernel_initializer=It))
        model.add(layers.Dense(256, activation='relu', kernel_initializer=It))
        model.add(layers.Dropout(.2))
        model.add(layers.Dense(4, activation='softmax', kernel_initializer=It))
    return model

def draw_img(history, num, round):
    # accuracy
    acc = history.history["accuracy"]
    epochs = range(1, len(acc)+1)
    val_acc = history.history["val_accuracy"]
    _,ax = plt.subplots(1,1)
    ax.plot(epochs, acc, "b-", label="Training Acc")
    ax.plot(epochs, val_acc, "r--", label="Validation Acc")
    ax.set_title("Training and Validation Accuracy")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.set_ylim(0,1)
    ax.legend(loc='lower right')
    plt.savefig('../result(tf1)/img/Accuracy client '+str(num)+' round '+str(round)+'.png')
    plt.close()

def result(model, num, round):
    test = pd.read_csv('../data/minmax-test.csv')
    size = pd.read_csv('../data/minmax-train.csv')
    x_test, y_test = preprocess(test, size)
    label = y_test.columns.tolist() # label for classification_report
    y_pre = np.argmax(model.predict(x_test), axis=1) # predict value
    y_act = LabelEncoder().fit_transform(test.iloc[:, 10]) # actual value
    #precision = precision_score(y_act, y_pre, average='macro')
    #recall = recall_score(y_act, y_pre, average='macro')
    #f1 = f1_score(y_act, y_pre, average='macro')
    report = classification_report(y_act, y_pre, target_names=label, digits=4, output_dict=True)
    pd.DataFrame(report).transpose().to_csv('../result(tf1)/table/result client '+str(num)+' round '+str(round)+'.csv', index=True)

def global_result(model, round):
    test = pd.read_csv('../data/minmax-test.csv')
    size = pd.read_csv('../data/minmax-train.csv')
    x_test, y_test = preprocess(test, size)
    label = y_test.columns.tolist() # label for classification_report
    y_pre = np.argmax(model.predict(x_test), axis=1) # predict value
    y_act = LabelEncoder().fit_transform(test.iloc[:, 10]) # actual value
    report = classification_report(y_act, y_pre, target_names=label, digits=4, output_dict=True)
    pd.DataFrame(report).transpose().to_csv('../result(tf1)/table/global-result round '+str(round)+'.csv', index=True)
    if drmat:
        draw_matrix(label, y_act, y_pre, round)

def draw_matrix(label, act, pred, round):
    # Sets the labels
    mat = confusion_matrix(act, pred)
    conf_matrix = pd.DataFrame(mat, index=label, columns=label)

    # plot size setting
    fig, ax = plt.subplots(figsize=(18, 15))
    sns.heatmap(conf_matrix, annot=True, fmt='d', annot_kws={"size": 8}, cmap="Blues")
    plt.ylabel('Actuals', fontsize=12)
    plt.xlabel('Predictions', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.savefig('../result(tf1)/img/Confusion-matrix global round '+str(round)+'.png')
    plt.close()
