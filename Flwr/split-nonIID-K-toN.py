import numpy as np
import pandas as pd
import heapq
import matplotlib.pyplot as plt

np.random.seed(42)

LOAD = True # GAN 200%

N = 5 # the number of clients
K = 3 # sampling the number of clients
ALPHA = 0.3 # alpha for dirichlet distribution

# Quantity-based label imbalance
def quantity_split(y, alpha, k):
    # quantity
    sy = sorted(set(y))
    dir = np.random.dirichlet([alpha]*k, len(sy))
    label_distribution = adjust(dir, K)
    # class-id
    class_idcs = []
    for l in sy:
        li = y.index[y==l].tolist()
        class_idcs.append(li)
    # client-id
    client_idcs = [[] for _ in range(k)]
    # client's class(label)-id
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]
    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
    return client_idcs

def adjust(a, K):
    for k in range(a.shape[0]):
        for i in range(N):
            a[k][i] = 1 if a[k][i] in heapq.nlargest(K, a[k]) else 0  
    b = a.T
    for i in range(b.shape[0]):
        print('client',i+1,':',int(sum(b[i])))

    return a/K

# using
if __name__ == "__main__":
    if LOAD:
        # Load dataset
        data = pd.read_csv("./data/minmax-train.csv")
        gens = pd.read_csv("./data/data-gen(rate=2.0).csv") # 生成集
        data = pd.concat([data, gens], axis=0).reset_index(drop=True) # 合併生成資料
    else:
        # Load dataset
        data = pd.read_csv("./data/minmax-train.csv")
    # label
    y = data.iloc[:, 10]
    # dirichlet distribution
    client_idcs = quantity_split(y, alpha=ALPHA, k=N)
    # print the number of datasets for 3 clients
    for i in range(N):
        print('client', i+1, ':', len(client_idcs[i]))
    # save
    for i in range(N):
        sp = data.iloc[client_idcs[i]]
        sp.to_csv('./data/data-client'+str(i+1)+'.csv', index=False)

    print('Split Data into Clients that is success.')

    # distribution
    plt.figure(figsize=(15,6))
    plt.hist([y[idc]for idc in client_idcs], stacked=True, 
            bins=np.arange(0-0.5, 24 + 1.5, 1),
            label=["Client {}".format(i+1) for i in range(N)], rwidth=0.5)
    plt.xticks(np.arange(25), range(1,26))
    plt.legend()
    plt.show()
