from sklearn.model_selection import train_test_split
import pandas as pd

load_gen = True # GAN 200%

N = 2

if __name__ == "__main__":
    if load_gen:
        # Load dataset
        t0 = pd.read_csv("./data/minmax-train.csv")
        gen = pd.read_csv("./data/data-gen(rate=2.0).csv") # 生成集
        t0 = pd.concat([t0, gen], axis=0).reset_index(drop=True) # 合併生成資料
    else:
        # Load dataset
        t0 = pd.read_csv("./data/minmax-train.csv")
    X_t0, y_t0 = t0.iloc[:, 0:10], t0.iloc[:, 10] # 分割資料

    # Split train set into N partitions and randomly use one for training.
    if   N == 2:
        X_t1, X_t2, y_t1, y_t2 = train_test_split(X_t0, y_t0, test_size=0.5, random_state=42)
    elif N == 3:
        X_t0, X_t1, y_t0, y_t1 = train_test_split(X_t0, y_t0, test_size=0.33333, random_state=42)
        X_t2, X_t3, y_t2, y_t3 = train_test_split(X_t0, y_t0, test_size=0.5, random_state=42)
    elif N == 4:
        #X_t0, X_t1, y_t0, y_t1 = train_test_split(X_t0, y_t0, test_size=0.5, random_state=42)
        #X_t3, X_t4, y_t3, y_t4 = train_test_split(X_t0, y_t0, test_size=0.5, random_state=42)
        #X_t1, X_t2, y_t1, y_t2 = train_test_split(X_t1, y_t1, test_size=0.5, random_state=42)
        X_t0, X_t1, y_t0, y_t1 = train_test_split(X_t0, y_t0, test_size=0.5, random_state=42)
        X_t2, X_t3, y_t2, y_t3 = train_test_split(X_t0, y_t0, test_size=0.4, random_state=42)
        X_t1, X_t4, y_t1, y_t4 = train_test_split(X_t1, y_t1, test_size=0.2, random_state=42)
    elif N == 5:
        X_t0, X_t5, y_t0, y_t5 = train_test_split(X_t0, y_t0, test_size=0.4, random_state=42)
        X_t0, X_t3, y_t0, y_t3 = train_test_split(X_t0, y_t0, test_size=0.3333, random_state=42)
        X_t1, X_t2, y_t1, y_t2 = train_test_split(X_t0, y_t0, test_size=0.5, random_state=42)
        X_t4, X_t5, y_t4, y_t5 = train_test_split(X_t5, y_t5, test_size=0.5, random_state=42)
    
    for i in range(N):
        i = str(i+1)
        data = pd.concat([eval('X_t'+i), eval('y_t'+i)], axis=1)
        data.to_csv('./data/data-client' + i + '.csv', index=False)
        print('The number of Client ' + i + ': ' + str(data.shape[0]) + ' support.')
    
    print('Split Data into Clients that is success.')