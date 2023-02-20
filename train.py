import os
import numpy as np
import pandas as pd
import torch as tf
from torch.utils import data
from torchvision import transforms,models
import matplotlib.pyplot as plt
from tqdm import tqdm
device="cuda" if tf.cuda.is_available() else "cpu"

"""label预处理"""
all_labels=pd.read_csv("data/all_labels.csv").values.reshape((-1,))
label_dic={}
for idx,label in enumerate(all_labels):
    label_dic[label]=idx

class Read_data(data.TensorDataset):
    def __init__(self, features, labels,trans=None):
        super().__init__()
        self.trans=trans
        self.features=features
        self.labels=labels
    def __getitem__(self, index):
        img=self.features[index]
        label=self.labels[index]
        if not self.trans==None:
            img=self.trans(img)
        return img,np.int64(label)
    def __len__(self):
        return self.labels.shape[0]
    

class Accumulator:
    """累加器"""
    def __init__(self,n):
        self.data=[0.0 for i in range(n)]

    def add(self,*args):#累加
        self.data=[a+float(b) for a,b in zip(self.data,args)]

    def reset(self):#归零
        self.data=[0.0 for i in range(len(self.data))]

    def __getitem__(self,idx):
        return self.data[idx]


def init_weight(m):
    """初始化网络权重"""
    if type(m)==tf.nn.Linear or type(m)==tf.nn.Conv2d:
        tf.nn.init.xavier_uniform_(m.weight)

def accuracy(y_hat,y):
    """计算一个batch的正确样本数"""
    y_hat=y_hat.argmax(axis=1)
    cmp=y_hat.type(y.dtype)==y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net,dataloader):
    """评估模型在指定数据集上的精度"""
    m=Accumulator(2)
    net.eval()
    with tf.no_grad():
        for X,y in dataloader:
            X=X.to(device); y=y.to(device)
            m.add(accuracy(net(X),y),y.numel())
    return m[0]/m[1]

def train_val_epoch(net,dataloader,loss_f,updater,mode):
    """
    单个epoch训练或预测
    返回平均预测损失函数值、平均预测精度
    """
    net.train()
    m=Accumulator(3)
    if mode=='train':
        print("training...")
        for X,y in tqdm(dataloader):
            X=X.to(device); y=y.to(device)
            y_hat=net(X)
            l=loss_f(y_hat,y)

            updater.zero_grad()
            l.backward()
            updater.step()

            with tf.no_grad():
                m.add(float(l.sum()),accuracy(y_hat,y),y.numel())
    elif mode=='val':
        print("valuating...")
        with tf.no_grad():
            for X,y in tqdm(dataloader):
                X=X.to(device); y=y.to(device)
                y_hat=net(X)
                l=loss_f(y_hat,y)
                m.add(float(l.sum()),accuracy(y_hat,y),y.numel())

    return m[0]/m[2],m[1]/m[2]

def train(net,train_dataloader,val_dataloader,loss_f,epoch_cnt,updater):
    """总训练函数"""
    if not os.path.exists("train_result"):
        os.mkdir("train_result")

    train_loss_lst,val_loss_lst,train_accuracy_lst,val_accracy_lst=[],[],[],[]
    print("training device: ",device)
    
    best_loss=1e9
    for epoch in range(1,epoch_cnt+1):
        print("epoch %d\n---------------------------------:"%(epoch))
        train_loss, train_accuracy=train_val_epoch(net,train_dataloader,loss_f,updater,mode='train')
        val_loss, val_accuracy=train_val_epoch(net,val_dataloader,loss_f,updater,mode='val')
        print("train_loss: %f, val_loss: %f, train_accuracy %f, val_accuracy %f"
        %(train_loss,val_loss,train_accuracy,val_accuracy))

        """保存当前模型，以及训练过程中验证效果最好的模型"""
        tf.save(net.state_dict(),os.path.join("train_result","last.pt"))
        if best_loss >val_loss:
            tf.save(net.state_dict(),os.path.join("train_result","best.pt"))
            best_loss=val_loss

        train_loss_lst.append(train_loss)
        val_loss_lst.append(val_loss)
        train_accuracy_lst.append(train_accuracy)
        val_accracy_lst.append(val_accuracy)

    plt.figure()
    plt.plot(list(range(1,epoch_cnt+1)),train_accuracy_lst,label='train accuracy')
    plt.plot(list(range(1,epoch_cnt+1)),val_accracy_lst,label='test accuracy')
    plt.title('accuracy')
    plt.legend()
    
    plt.figure()
    plt.plot(list(range(1,epoch_cnt+1)),train_loss_lst,label='train loss')
    plt.plot(list(range(1,epoch_cnt+1)),val_loss_lst,label='val loss')
    plt.title('loss')
    plt.legend()

def val(net,val_dataloader,loss_f):
    loss,acc=train_val_epoch(net,val_dataloader,loss_f)
    print("val_loss: %f, val_accuracy %f"
        %(loss,acc))


if __name__=='__main__':
    """初始化"""
    train_data_rate=0.8
    batchsize=128
    epoch_cnt=20
    lr=0.005
    # resize=96       #2060显卡显存装不下224x224的#

    """导入数据"""
    all_data=pd.read_csv("data/train.csv").values

    print(all_data.shape)
    """数据增强"""
    train_trans=[
        transforms.ToTensor(),
        # transforms.Resize((resize,resize)),
        transforms.ColorJitter(brightness=0.2,contrast=0.5,saturation=0.5,hue=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ]
    train_trans=transforms.Compose(train_trans) 
    val_trans=[transforms.ToTensor()]#,transforms.Resize((resize,resize))]
    val_trans=transforms.Compose(val_trans)

    features=np.array([np.array(plt.imread(os.path.join("data",img))) for img in all_data[:,0]])
    labels=np.array([label_dic[label] for label in all_data[:,1]])
    features_cnt=int(labels.shape[0]*train_data_rate)
    
    train_features=features[0:features_cnt]
    val_features=features[features_cnt+1:]
    train_labels=labels[0:features_cnt]
    val_labels=labels[features_cnt+1:]

    train_dataloader=data.DataLoader(Read_data(train_features,train_labels,train_trans),batch_size=batchsize,drop_last=True,num_workers=12,shuffle=True)
    val_dataloader=data.DataLoader(Read_data(val_features,val_labels,val_trans),batch_size=batchsize,drop_last=True,num_workers=12)

    """网络模型"""
    net=models.resnet50(num_classes=176)
    net.conv1=tf.nn.Conv2d(3,64,kernel_size=7,padding=3,stride=2)
    net.apply(init_weight)
    net.to(device)
    net.load_state_dict(tf.load("train_result/best.pt"))
    
    # for X,y in train_dataloader:
    #     print(X)
    #     print(net(X).shape)
    #     print(net(X).dtype)
    #     break
    loss=tf.nn.CrossEntropyLoss()
    updater=tf.optim.Adam(net.parameters(),lr=lr)
    train(net,train_dataloader,val_dataloader,loss,epoch_cnt,updater)
    plt.show()
    # val(net,val_dataloader,loss)
    