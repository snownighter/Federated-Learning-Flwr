
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.pyplot import MultipleLocator
# from tensorflow.python.keras.m
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras import optimizers
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import FeatureHasher
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, precision_score, f1_score, recall_score
from sklearn.utils import shuffle

import time

rate_list = [0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 1.8, 2.0] # 生成比例

def preprocess(data):
    str_name = ['src_ip', 'dst_ip', 'server_port', 'prot']
    num_name = ['p_count', 'b_count', 'max_size', 'min_size', 'abyte_count', 'sbyte_count']
    # lab_name = ['']

    # data.columns = str_name + num_name + lab_name # 定義column
    X_str, Y_lab = data.loc[:, str_name], data.iloc[:, 10] # 分割資料
    
    Y_lab = pd.get_dummies(Y_lab, prefix_sep='') # One-hot編碼

    vector_size = np.zeros(4, dtype=np.int32) # 設置hash空間
    for i in range(4):
        vector_size[i] = len(train.iloc[:,i].unique()) # 限定train大小 (使train和test相同大小)

    ct = ColumnTransformer([
        ('src_ip',      FeatureHasher(n_features=vector_size[0], input_type='string'), 0),
        ('dst_ip',      FeatureHasher(n_features=vector_size[1], input_type='string'), 1),
        ('server_port', FeatureHasher(n_features=vector_size[2], input_type='string'), 2),
        ('proto',       FeatureHasher(n_features=vector_size[3], input_type='string'), 3)])
    str_t = X_str.iloc[:,0:4].astype('str') 
    X_str = pd.DataFrame(ct.fit_transform(str_t).toarray()) # hash 之 One-hot編碼
    
    return pd.concat([X_str, data.loc[:, num_name]], axis=1), Y_lab # 合併資料

def draw_img(history):
    # 顯示訓練和驗證損失
    loss = history.history["loss"]
    epochs = range(1, len(loss)+1)
    val_loss = history.history["val_loss"]
    plt.figure()
    plt.plot(epochs, loss, "b-", label="Training Loss")
    plt.plot(epochs, val_loss, "r--", label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.gca().yaxis.set_major_locator(MultipleLocator(.1))
    plt.ylim(0,1)
    plt.legend()
    plt.savefig('./re-MLP/run5/Loss-for-MLP(rate=' + rate + ').png')
    plt.close()
    
    # 顯示訓練和驗證準確度
    acc = history.history["accuracy"]
    epochs = range(1, len(acc)+1)
    val_acc = history.history["val_accuracy"]
    plt.figure()
    plt.plot(epochs, acc, "b-", label="Training Acc")
    plt.plot(epochs, val_acc, "r--", label="Validation Acc")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.gca().yaxis.set_major_locator(MultipleLocator(.1))
    plt.ylim(0,1)
    plt.legend()
    plt.savefig('./re-MLP/run5/Accuracy-for-MLP(rate=' + rate + ').png')
    plt.close()
    
def draw_matrix(Y_test, Y_target, Y_pred):
    # 顯示混淆矩陣
    label = Y_test.columns.tolist()
    # Sets the labels
    mat = confusion_matrix(Y_target, Y_pred)
    conf_matrix = pd.DataFrame(mat, index=label, columns=label)
    
    # plot size setting
    fig, ax = plt.subplots(figsize = (18, 15))
    sns.heatmap(conf_matrix, annot=True, fmt='d', annot_kws={"size": 8}, cmap="Blues")
    plt.ylabel('Actuals', fontsize=12)
    plt.xlabel('Predictions', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.savefig('./re-MLP/run5/Confusion-matrix(rate=' + rate + ').png')
    plt.close()

def trainMLP(retrain, test):
    # 資料預處理
    X_retrain, Y_retrain = preprocess(retrain)
    X_test, Y_test = preprocess(test)
    Y_target = test.iloc[:, 10]

    # 定義模型
    model = Sequential()
    model.add(Dense(256, input_dim=X_retrain.shape[1], activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(.2))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(.2))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(.2))
    model.add(Dense(25, activation="softmax"))
    model.summary()   # 顯示模型摘要資訊

    # # 載入模型
    # from keras.models import load_model
    
    # model = Sequential()
    # model = load_model("MLP(rate=1.0).h5")
    
    # 編譯模型
    adam = optimizers.Adam(learning_rate=0.01)
    model.compile(loss="categorical_crossentropy", optimizer=adam,
                  metrics=["accuracy"])

    # 訓練模型
    history = model.fit(X_retrain, Y_retrain, validation_split=0.2, 
                        epochs=100, batch_size=4096, verbose=1)
    
    # 儲存模型結構和權重
    model.save("./re-MLP/run5/MLP(rate=" + rate + ").h5")

    # 評估模型
    loss, accuracy = model.evaluate(X_retrain, Y_retrain)
    print("訓練資料集的準確度 = {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, Y_test)
    print("測試資料集的準確度 = {:.4f}".format(accuracy))

    # 繪製圖表
    draw_img(history)

    # 計算分類的預測值
    Y_pred = np.argmax(model.predict(X_test), axis=1)
    
    labelencoder = LabelEncoder()
    Y_target = labelencoder.fit_transform(Y_target)
    
    # 繪製混淆矩陣
    draw_matrix(Y_test, Y_target, Y_pred)

    # 顯示結果
    precision = precision_score(Y_target, Y_pred, average='macro')
    recall = recall_score(Y_target, Y_pred, average='macro')
    f1 = f1_score(Y_target, Y_pred, average='macro')
    
    print('Macro precision = {:.4f}'.format(precision))
    print('Macro recall    = {:.4f}'.format(recall))
    print('Macro f1-score  = {:.4f}'.format(f1))
    
    label = Y_test.columns.tolist()
    
    from sklearn.metrics import classification_report
    print(classification_report(Y_target, Y_pred, target_names=label, digits=4))
    report = classification_report(Y_target, Y_pred, target_names=label, digits=4, output_dict=True)
    pd.DataFrame(report).transpose().to_csv('./re-MLP/run5/result(rate=' + rate + ').csv', index=True)

start_time = time.ctime(time.time()) # 開始

# 各生成數訓練
for i in range(len(rate_list)):
    rate = str(rate_list[i]) # 生成數
    # 載入
    train = pd.read_csv("./data/minmax-train.csv")
    gen = pd.read_csv("./data/data-gen(rate=" + rate + ").csv") # 生成集
    test = pd.read_csv("./data/minmax-test.csv")
    # 合併
    retrain = pd.concat([train, gen], axis=0).reset_index(drop=True) # 合併生成資料
    retrain = shuffle(retrain) # 打亂資料
    
    trainMLP(retrain, test) # 訓練

end_time = time.ctime(time.time()) # 結束
print(start_time + '\n' + end_time)

