import torch
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
import test_sp as sp

torch.manual_seed(42)

if __name__ == "__main__":

    N_CLIENTS = 10 
    DIRICHLET_ALPHA = 1.0

    train_data = datasets.EMNIST(root=".", split="byclass", download=True, train=True)
    test_data = datasets.EMNIST(root=".", split="byclass", download=True, train=False)
    n_channels = 1


    input_sz, num_cls = train_data.data[0].shape[0],  len(train_data.classes) # 28, 62


    train_labels = np.array(train_data.targets) # 597932

    # index = 697932, data-size = (28,28), class-num = 62

    # 我们让每个client不同label的样本数量不同，以此做到Non-IID划分
    client_idcs = sp.dirichlet_split_noniid(train_labels, alpha=DIRICHLET_ALPHA, n_clients=N_CLIENTS)


    for i in range(len(client_idcs)):
        print('len',i,':',len(client_idcs[i]))
    print(client_idcs[0])

    # 展示不同client的不同label的数据分布
    #plt.figure(figsize=(20,3))
    #plt.hist([train_labels[idc]for idc in client_idcs], stacked=True, 
    #        bins=np.arange(min(train_labels)-0.5, max(train_labels) + 1.5, 1),
    #        label=["Client {}".format(i) for i in range(N_CLIENTS)], rwidth=0.5)
    #plt.xticks(np.arange(num_cls), train_data.classes)
    #plt.legend()
    #plt.show()