
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.pyplot import MultipleLocator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import optimizers
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction import FeatureHasher
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, precision_score, f1_score, recall_score

seed = 7
np.random.seed(seed)

# 載入141-7的訓練和測試資料集
df_train = pd.read_csv("./141-7_train.csv").values
df_test = pd.read_csv("./141-7_test.csv").values

# 打亂資料
np.random.shuffle(df_train)
np.random.shuffle(df_test)

# 分割成特徵資料和標籤資料
X_train_oh, X_train_s, Y_train = df_train[:, 0:4], df_train[:, 4:10], df_train[:, 10]
X_test_oh, X_test_s, Y_test = df_test[:, 0:4], df_test[:, 4:10], df_test[:, 10]

Y_target = pd.DataFrame(Y_test)

# MinMax標準化
X_train_s = pd.DataFrame(
    MinMaxScaler(feature_range=(0, 1)).fit(X_train_s).transform(X_train_s))
X_test_s = pd.DataFrame(
    MinMaxScaler(feature_range=(0, 1)).fit(X_test_s).transform(X_test_s))

# One-hot編碼
Y_train = pd.get_dummies(Y_train)
Y_test = pd.get_dummies(Y_test)

vector_size = np.zeros(4, dtype=np.int32)

# train
for i in range(4):
    vector_size[i] = len(pd.DataFrame(df_train)[i].unique())

ct = ColumnTransformer([
    ('src_ip',      FeatureHasher(n_features=vector_size[0], input_type='string'), 0),
    ('dst_ip',      FeatureHasher(n_features=vector_size[1], input_type='string'), 1),
    ('server_port', FeatureHasher(n_features=vector_size[2], input_type='string'), 2),
    ('proto',       FeatureHasher(n_features=vector_size[3], input_type='string'), 3)])
str_t = pd.DataFrame(X_train_oh)[[0,1,2,3]].astype('str')
X_train_oh = pd.DataFrame(ct.fit_transform(str_t).toarray())

X_train = pd.concat([X_train_oh, X_train_s], axis=1)

# test
for i in range(4):
    vector_size[i] = len(pd.DataFrame(df_test)[i].unique())

str_t = pd.DataFrame(X_test_oh)[[0,1,2,3]].astype('str')
X_test_oh = pd.DataFrame(ct.fit_transform(str_t).toarray())

X_test = pd.concat([X_test_oh, X_test_s], axis=1)

# 定義模型
model = Sequential()
model.add(Dense(256, input_dim=X_train.shape[1], activation="relu"))
model.add(Dense(256, activation="relu"))
model.add(Dropout(.2))
model.add(Dense(256, activation="relu"))
model.add(Dropout(.2))
model.add(Dense(25, activation="softmax"))
model.summary()   # 顯示模型摘要資訊

# # 載入模型
# from keras.models import load_model

# model = Sequential()
# model = load_model("MLP-05.h5")

# 編譯模型
adam = optimizers.Adam(learning_rate=0.01)
model.compile(loss="categorical_crossentropy", optimizer=adam,
              metrics=["accuracy"])

# 訓練模型
history = model.fit(X_train, Y_train, validation_split=0.2, 
          epochs=100, batch_size=4096, verbose=1)

# 儲存模型結構和權重
model.save("MLP-05-ex.h5")

# 評估模型
loss, accuracy = model.evaluate(X_train, Y_train)
print("訓練資料集的準確度 = {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, Y_test)
print("測試資料集的準確度 = {:.4f}".format(accuracy))

# 顯示圖表來分析模型的訓練過程

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
plt.savefig('./MLP-05-img/Loss-for-MLP-ex.png')
plt.show()

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
plt.savefig('./MLP-05-img/Accuracy-for-MLP-ex.png')
plt.show()

# 計算分類的預測值
Y_pred = np.argmax(model.predict(X_test), axis=1)

labelencoder = LabelEncoder()
Y_target = labelencoder.fit_transform(Y_target)

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
plt.savefig('./MLP-05-img/Confusion-matrix-ex.png')
plt.show()

# 舊版本

# mat_con = confusion_matrix(Y_target, Y_pred)

# # Setting the attributes
# fig, px = plt.subplots(figsize=(10, 10))
# px.matshow(mat_con, cmap=plt.cm.YlOrRd, alpha=0.5)
# for m in range(mat_con.shape[0]):
#     for n in range(mat_con.shape[1]):
#         px.text(x=m,y=n,s=mat_con[m, n], va='center', ha='center', size='xx-large', fontsize=10)

# # Sets the labels
# plt.xlabel('Predictions', fontsize=10)
# plt.ylabel('Actuals', fontsize=10)
# plt.title('Confusion Matrix', fontsize=10)
# plt.savefig('./MLP-03-img/Confusion-matrix.png')
# plt.show()

precision = precision_score(Y_target, Y_pred, average='macro')
recall = recall_score(Y_target, Y_pred, average='macro')
f1 = f1_score(Y_target, Y_pred, average='macro')

print('Macro precision = {:.4f}'.format(precision))
print('Macro recall    = {:.4f}'.format(recall))
print('Macro f1-score  = {:.4f}'.format(f1))

from sklearn.metrics import classification_report
print(classification_report(Y_target, Y_pred, target_names=label, digits=4))
report = classification_report(Y_target, Y_pred, target_names=label, digits=4, output_dict=True)
pd.DataFrame(report).transpose().to_csv('./MLP-05-img/result-ex.csv', index=True)

