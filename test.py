import os
import numpy as np
import pandas as pd
import torch as tf
from torch.utils import data
from torchvision import transforms,models
import matplotlib.pyplot as plt
from tqdm import tqdm
device="cuda" if tf.cuda.is_available() else "cpu"

class Read_data(data.TensorDataset):
    def __init__(self, features,trans=None):
        super().__init__()
        self.trans=trans
        self.features=features
    def __getitem__(self, index):
        img=self.features[index]
        if not self.trans==None:
            img=self.trans(img)
        # img.to(device); label.to(device)
        return img
    def __len__(self):
        return self.features.shape[0]

all_labels=pd.read_csv("data/all_labels.csv").values.reshape((-1,))
label_dic={}; idx_dic={}
for idx,label in enumerate(all_labels):
    label_dic[label]=idx
    idx_dic[idx]=label



def test(net,dataloader):
    pred=None
    print("testing......")
    with tf.no_grad():
        for X in tqdm(dataloader):
            X=X.to(device)
            y_hat=net(X).argmax(1)
            if pred==None:
                pred=y_hat
            else:
                pred=tf.cat([pred,y_hat])

    return pred


if __name__=='__main__':
    batchsize=128

    test_data=pd.read_csv("data/test.csv").values
    features=np.array([np.array(plt.imread(os.path.join("data",img))) for img in test_data[:,0]])
    test_trans=[transforms.ToTensor()]
    test_trans=transforms.Compose(test_trans)
    test_dataloader=data.DataLoader(Read_data(features,test_trans),batch_size=batchsize,num_workers=12)

    """网络模型"""
    net=models.resnet50(num_classes=176)
    net.conv1=tf.nn.Conv2d(3,64,kernel_size=7,padding=3,stride=2)
    net.to(device)
    net.load_state_dict(tf.load("train_result/best.pt"))
    """预测"""
    pred=np.array(test(net,test_dataloader).cpu())
    labels=np.array([idx_dic[i] for i in pred]).reshape((-1,1))
    result=np.hstack((test_data,labels))
    """保存结果"""
    if not os.path.exists("result"):
        os.mkdir("result")
    pd.DataFrame(result).to_csv("result/result.csv",index=False,header=['image','label'])

