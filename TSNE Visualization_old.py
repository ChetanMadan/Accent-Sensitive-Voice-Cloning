
from __future__ import print_function
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.manifold import TSNE
import time
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


import librosa 
import librosa.display
import IPython
import pickle 
import numpy as np
import scipy 
import tensorflow as tf 
from sklearn.externals import joblib
from tensorflow.python.client import device_lib
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Sequential
from keras import backend as K
from keras.optimizers import Adam
from tensorflow.saved_model import simple_save


import os
import glob
import random
import time
from utils import mfccs_and_spec
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from random import shuffle
from hparam import hparam as hp
from torch.functional import F
import torch.nn as nn
from torch.utils.data import Dataset
from torch.autograd import Variable, Function
import torch.optim as optim
from data_load import TripletSpeakerDataset
from speech_embedder_net import SpeechEmbedder, GE2ELoss, get_centroids, get_cossim

# In[50]:


def make_accent_model():
    model = Sequential()

    model.add(Dense(50, input_shape = (14976,), activation = 'relu',name='input'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(75, activation = 'tanh', name='h1'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(100,  activation = 'tanh',name = 'h2'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(100,  activation = 'tanh', name = 'h3'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(100,  activation = 'tanh', name = 'h4'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(100,  activation = 'tanh', name = 'h5'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(100,  activation = 'tanh', name ='h6'))
    model.add(BatchNormalization())

    model.add(Dense(3,  activation = 'softmax', name = 'output'))
    return model


##Mel spectrograms and MFCCs for Indians 
def get_accent_mfccs(path):
    x, _ = librosa.load(path)
    x, mf = mfcc_for_accent(x)
    mf = mf.flatten()
    #print(type(x))
    # l.append(librosa.feature.mfcc(x, sr=f))
    
    dat = np.reshape(mf, (1,mf.shape[0]))
    return x, dat

class PairwiseDistance(Function):
    def __init__(self, p):
        super(PairwiseDistance, self).__init__()
        self.norm = p

    def forward(self, x1, x2):
        #print("above assert: ",x1, x2)
        assert x1.size() == x2.size()
        eps = 1e-4 / x1.size(1)
        diff = torch.abs(x1 - x2)
        out = torch.pow(diff, self.norm).sum(dim=1)
        return torch.pow(out + eps, 1. / self.norm)

class TripletLoss(Function):
    """
    Triplet loss function.
    """
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.pdist = PairwiseDistance(2)  # norm 2

    def forward(self, anchor, positive, negative):
        d_p = self.pdist.forward(torch.from_numpy(anchor), positive)
        d_n = self.pdist.forward(torch.from_numpy(anchor), torch.from_numpy(negative))

        dist_hinge = torch.clamp(self.margin + d_p - d_n, min=0.0)
        loss = torch.mean(dist_hinge)
        return loss

def run_speaker_encoder(mfccs, embedder_net):
    mel_db = torch.transpose(mfccs, 1,2)
    #mel_db = torch.from_numpy(mel_db)
    #print(mel_db.shape)
    out = embedder_net(mel_db.to('cuda'))
    return out

def get_both_embs(path):
        #for path in enumerate(train_dataset):
        _, accent_mfccs = get_accent_mfccs(path)
        speaker_mfccs = get_spectrogram_tisv(path)
        #print(accent_mfccs.shape)
        #print(speaker_mfccs.shape)
        speaker_mfccs = Variable(torch.from_numpy(speaker_mfccs)).to('cuda')
        speaker_emb = run_speaker_encoder(speaker_mfccs, embedder_net)
        #print(speaker_emb.shape)
        accent_emb = accent_net.predict(accent_mfccs)
        #print(accent_emb.shape)
        return speaker_emb.cpu().detach().numpy(), accent_emb


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.inp1 = nn.Linear(in_features=256, out_features=100)
        self.inp2 = nn.Linear(in_features=100, out_features=100)
        self.b_n = nn.BatchNorm1d(128)
        
        self.drop_out = nn.Dropout(0.25)
        self.l1 = nn.Linear(200, 128)
        self.l2 = nn.Linear(128,128)
        self.out_acc = nn.Linear(128, 100)
        self.out_speak = nn.Linear(128,256)
        
    def forward(self, *args):
        speaker_emb, acc_emb = args[0], args[1]
        x1 = F.relu(self.inp1(speaker_emb))
        
        x2 = F.relu(self.inp2(x1))
        #print(x1.shape, x2.shape, speaker_emb.shape, acc_emb.shape)
        out = torch.cat((x1, x2), axis=1)
        out = self.l1(out)
        out = F.relu(out)
        out = F.relu(self.l2(out))
        
        mf_acc = self.out_acc(out)
        mf_sp = self.out_speak(out)
        return F.relu(mf_sp), F.relu(mf_acc)

def preprocess_tsne(model, path):
    tsne = TSNE(verbose=1)
    utters = []
    files = glob.glob(path)
    speakers = []
    X1=[]
    for i,file in enumerate(files):
        
        if i%500!=0:
            continue
        try:
            pre_positive_speaker_embs, pre_positive_accent_embs = get_both_embs(file)
        except:
            print('continuing')
            continue
        speaker = file.split('/')[-2]
        pre_positive_speaker_embs, pre_positive_accent_embs = Variable(torch.from_numpy(pre_positive_speaker_embs)).to('cpu'), Variable(torch.from_numpy(pre_positive_accent_embs)).to('cpu')
        utt, _ = ae.forward(pre_positive_speaker_embs, pre_positive_accent_embs)
        print(i)
        utters.append(utt)
        speakers.append(speaker)
        X1.append(get_spectrogram_tisv(file))
    arr = [t.detach().numpy() for t in utters]
    y = np.asarray(speakers)
    print(speakers)
    print(np.asarray(arr).shape)
    X = np.squeeze(np.asarray(arr), 1)
    return X, y, X1


def tsne_visualize(X, y):
    feat_cols = ['pixel'+str(i) for i in range(X.shape[1])]
    df = pd.DataFrame(X,columns=feat_cols)
    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))
    print('Size of the dataframe: {}'.format(df.shape))
    
    np.random.seed(42)
    rndperm = np.random.permutation(df.shape[0])
    
    N = 10000
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(df[feat_cols].values)
    df['pca-one'] = pca_result[:,0]
    df['pca-two'] = pca_result[:,1] 
    df['pca-three'] = pca_result[:,2]
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    plt.figure(figsize=(16,10))
    sns_plot = sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="y",
        palette=sns.color_palette("hls", 14),
        data=df.loc[rndperm,:],
        legend="full",
        alpha=1.0
    )
    sns_plot.figure.savefig('fig.png')
    #plt.savefig("out.png")
    df_subset = df.loc[rndperm[:N],:].copy()
    data_subset = df_subset[feat_cols].values
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=10000)
    print(X.shape)
    tsne_results = tsne.fit_transform(pd.DataFrame(np.asarray(X)))
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    df_subset['tsne-2d-one'] = tsne_results[:,0]
    df_subset['tsne-2d-two'] = tsne_results[:,1]
    plt.figure(figsize=(16,10))


embedder_net = SpeechEmbedder().to('cuda')
embedder_net.load_state_dict(torch.load(hp.model.model_path))
accent_net = make_accent_model()
accent_net.load_weights('accent_block_with_scottish.h5')
accent_net.pop()
dataset = TripletSpeakerDataset()
data_loader = torch.utils.data.DataLoader(dataset, drop_last=True)

#wav, sr = librosa.load(file_name)

ae = AutoEncoder().to('cpu')
ae.load_state_dict(torch.load("./new_checkpoints/ckpt_epoch_5_batch_id_8000.pth")['state_dict'])

#ae.load_state_dict(torch.load("./new_checkpoints/ckpt_epoch_10900_batch_id_10900.pth")['state_dict'])
print(hp.data.train_path)
X, y ,specs= preprocess_tsne(ae, hp.data.train_path)


# In[51]:


tsne_visualize(X, y)


# In[39]:


tsne_visualize(np.asarray(specs).reshape(np.asarray(specs).shape[0], -1), y)


# In[40]:


X1.reshape(X1.shape[0], -1).shape


# In[41]:


y.shape


# In[42]:


tsne_visualize(X, y)


# In[43]:


tsne_visualize(X, y)


# In[44]:


tsne_visualize(X, y)


# In[49]:


tsne_visualize(X, y)


# In[59]:


pca = PCA(n_components=3)


# In[61]:





# In[ ]:





# In[64]:


import torchviz


# In[68]:


pre_positive_speaker_embs, pre_positive_accent_embs = get_both_embs(file_name)
pre_positive_speaker_embs, pre_positive_accent_embs = Variable(torch.from_numpy(pre_positive_speaker_embs)).to('cpu'), Variable(torch.from_numpy(pre_positive_accent_embs)).to('cpu')


# In[69]:


pre_positive_accent_embs.shape


# In[70]:


pre_positive_speaker_embs.shape


# In[10]:


from pytorch2keras.converter import pytorch_to_keras


# In[33]:


x1 = torch.zeros(1,100, dtype=torch.float, requires_grad=True)
x2 = torch.zeros(1,256, dtype=torch.float, requires_grad=True)
ae = AutoEncoder()
out = ae.forward(x2,x1)
torchviz.make_dot(out)


# In[90]:


pytorch_total_params = sum(p.numel() for p in ae.parameters() if p.requires_grad)


# In[91]:


pytorch_total_params


# In[32]:


import onnx
import torchviz


# In[28]:


x1 = torch.randn(batch_size, 1, 100, requires_grad=True)
x1 = torch.randn(batch_size, 1, 256, requires_grad=True)


# In[34]:


torch.onnx.export(ae,
          (x1,x2),
          'test.onnx',
          export_params=True)


# In[48]:




# In[ ]:




