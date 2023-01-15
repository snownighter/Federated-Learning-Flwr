import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

if __name__ == "__main__":
    for i in range(5):
        y = pd.read_csv('./data/data-client'+str(i+1)+'.csv').iloc[:, 10]
        report = classification_report(y, y, target_names=sorted(set(y)), digits=4, output_dict=True)
        df = pd.DataFrame(report).transpose().astype(int)['support'].iloc[0:len(sorted(set(y)))]
        data = df if i==0 else pd.concat([data, df], axis=1)
    data.columns = ['client1', 'client2', 'client3', 'client4', 'client5']
    data = data.fillna(0).astype(int).sort_index()
    print(data)
    data.to_csv('./data/support.csv')
