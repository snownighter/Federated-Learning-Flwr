import numpy as np

np.random.seed(42)
def  dirichlet_split_noniid(train_labels, alpha, n_clients):
    n_classes = train_labels.max()+1 # 62
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    # (K, N)的类别标签分布矩阵X，记录每个client占有每个类别的多少 (62,10)

    print('train_labels:', train_labels)

    # print(label_distribution)
    print(label_distribution.shape)

    class_idcs = [np.argwhere(train_labels==y).flatten() 
           for y in range(n_classes)]
    # 记录每个K个类别对应的样本下标
    print('class:', len(class_idcs[0]))
 
    client_idcs = [[] for _ in range(n_clients)]
    print('client:', client_idcs)
    cc = 0
    for c, fracs in zip(class_idcs, label_distribution):
        if cc == 0: print('zip:', c, fracs)
        print('zip len',cc,':', len(c), len(fracs))
        cc += 1
    print('cc:', cc)
    # 记录N个client分别对应样本集合的索引
    for c, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例将类别为k的样本划分为了N个子集
        # for i, idcs 为遍历第i个client对应样本集合的索引
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]

    print('cid:', len(client_idcs[0]))
    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
    print('cid:', len(client_idcs[0]))

    print(len(client_idcs))
    print(client_idcs[0].shape)
    print(client_idcs[1].shape)
  
    return client_idcs