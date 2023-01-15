import numpy as np
import pandas as pd
np.random.seed(42)


data = pd.read_csv('./data/minmax-test.csv')
X, y = data.iloc[:, 0:10], data.iloc[:, 10]
print('X:', X.shape, 'y:', y.shape)
sy = sorted(set(y))
print('sy:', len(sy))

label_distribution = np.random.dirichlet([1.0]*3, len(sy))

# for l in sy: print(l)
class_idcs = []
for l in sy:
    li = y.index[y==l].tolist()
    class_idcs.append(li)

# print(len(class_idcs))
# print(class_idcs[0])
# for i in range(len(sy)): print('len',i,':',len(class_idcs[i]))

client_idcs = [[] for _ in range(3)]
# cc = 0
# for c, fracs in zip(class_idcs, label_distribution):
#     if cc == 0: print('zip:', c, fracs)
#     print('zip len',cc,':', len(c), len(fracs))
#     cc += 1
# print('cc:', cc)
for c, fracs in zip(class_idcs, label_distribution):
    for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
        client_idcs[i] += [idcs]
client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

for i in range(3):
    print('client',i,':',len(client_idcs[i]))

for i in range(3):
    sp = data.iloc[client_idcs[i]]
    # if i==0: sp.to_csv('test_'+str(i)+'.csv')