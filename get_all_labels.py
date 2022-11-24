import pandas as pd
import numpy as np

"""获取树叶的所有种类，保存在 data\all_labels.csv 中"""

if __name__=='__main__':
    data=pd.read_csv(r"data\train.csv").values[:,1]
    labels=np.unique(data)
    pd.DataFrame(labels.reshape((-1,1))).to_csv(r"data\all_labels.csv",index=False)
