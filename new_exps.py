#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


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


# In[3]:


from utils import mfcc_for_accent
from data_preprocess import get_spectrogram_tisv

file_name = "/home/dexter/Desktop/projects/Mini/data/VCTK-Corpus/wav48_silence_trimmed/p225/p225_001_mic2.flac"
# In[4]:


config = tf.compat.v1.ConfigProto(allow_soft_placement = True,
                        device_count = {'CPU' : 6,
                                       'GPU' : 0})


# In[5]:


session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


# In[6]:


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


# In[ ]:





# In[7]:


## Mel spectrograms and MFCCs for Indians 
def get_accent_mfccs(path):
    x, _ = librosa.load(path)
    x, mf = mfcc_for_accent(x)
    mf = mf.flatten()
    #print(type(x))
    # l.append(librosa.feature.mfcc(x, sr=f))
    
    dat = np.reshape(mf, (1,mf.shape[0]))
    return x, dat


# In[8]:



embedder_net = SpeechEmbedder().to('cuda')
embedder_net.load_state_dict(torch.load(hp.model.model_path))
accent_net = make_accent_model()
accent_net.load_weights('accent_block_with_scottish.h5')
accent_net.pop()


#wav, sr = librosa.load(file_name)

def run_speaker_encoder(mfccs, embedder_net):
    mel_db = torch.transpose(mfccs, 1,2)
    #mel_db = torch.from_numpy(mel_db)
    #print(mel_db.shape)
    out = embedder_net(mel_db.to('cuda'))
    return out


# In[9]:


from keras.utils import plot_model


# In[13]:


#plot_model(accent_net, to_file='accent.png')


# In[10]:


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


# In[ ]:





# In[11]:


accent_net.summary()


# In[12]:


dataset = TripletSpeakerDataset()
data_loader = torch.utils.data.DataLoader(dataset, drop_last=True)


# In[13]:


#for batch_idx, (anchor_utter_path, positive_utter_path, negative_utter_path, accent_negative_path, speaker_positive_id, speaker_negative_id, speaker_anchor_id, positive_accent, negative_accent) in enumerate(data_loader):
#    print(anchor_utter_path, positive_utter_path, negative_utter_path, accent_negative_path, speaker_positive_id, speaker_negative_id, speaker_anchor_id, positive_accent, negative_accent)


# In[ ]:


#speaker_positive_id[0], speaker_negative_id, speaker_anchor_id, positive_accent, negative_accent


# In[ ]:


#anchor_utter_path, positive_utter_path, negative_utter_path, accent_negative_path


# In[14]:


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


# In[15]:


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


# In[16]:


def train():
    device = hp.device
    dataset = TripletSpeakerDataset()
    data_loader = torch.utils.data.DataLoader(dataset, drop_last=True)
    ae = AutoEncoder().to('cuda')
    writer = SummaryWriter('triplet_loss_logs')
    optimizer = optim.Adam(ae.parameters())
    optimizer.zero_grad()
    #ae.load_state_dict(torch.load(''))
    trip_loss = TripletLoss()
    total_loss = 0
    for epoch in range(20):
        for batch_id, (anchor_utter_path, positive_utter_path, negative_utter_path, accent_negative_path, speaker_positive_id, speaker_negative_id, speaker_anchor_id, positive_accent, negative_accent) in tqdm(enumerate(data_loader), total = len(data_loader)):
            anchor_utter_path, positive_utter_path, negative_utter_path = anchor_utter_path[0], positive_utter_path[0], negative_utter_path[0]
            accent_negative_path, speaker_positive_id, speaker_negative_id = accent_negative_path[0], speaker_positive_id[0], speaker_negative_id[0]
            speaker_anchor_id, positive_accent, negative_accent = speaker_anchor_id[0], positive_accent[0], negative_accent[0]
            
            try:
                pre_positive_speaker_embs, pre_positive_accent_embs = get_both_embs(positive_utter_path)
                pre_positive_speaker_embs, pre_positive_accent_embs = Variable(torch.from_numpy(pre_positive_speaker_embs)).to('cuda'), Variable(torch.from_numpy(pre_positive_accent_embs)).to('cuda')
                anchor_utter_path
                anchor_speaker_embs, anchor_accent_embs = get_both_embs(anchor_utter_path)
                negative_speaker_embs, _ = get_both_embs(negative_utter_path)
                _, negative_accent_embs = get_both_embs(accent_negative_path)
            except ValueError:
                print("ValueError")
                continue
            except:
                continue
            positive_speaker_embs, positive_accent_embs = ae.forward(pre_positive_speaker_embs, pre_positive_accent_embs)
            positive_speaker_embs, positive_accent_embs = positive_speaker_embs.to('cpu'), positive_accent_embs.to('cpu')
            optimizer.zero_grad()
            #print(anchor_accent_embs.shape)
            accent_trip_loss = trip_loss.forward(anchor_accent_embs, positive_accent_embs, negative_accent_embs)
            speaker_trip_loss = trip_loss.forward(anchor_speaker_embs, positive_speaker_embs, negative_speaker_embs)
            criterion = accent_trip_loss + speaker_trip_loss
            total_loss += criterion
            
            
            
            criterion.backward()
            optimizer.step()
            writer.add_scalar('Loss', criterion, epoch*len(data_loader)+1)
            
            if (batch_id + 1) % 5 == 0:
                mesg = "{}\tEpoch:{},Iteration:{}\tLoss:{}\t\n".format(time.ctime(), epoch+1,
                        batch_id+1, criterion)
                print(mesg)
                if hp.train.log_file is not None:
                    with open(hp.train.log_file,'a') as f:
                        f.write(mesg)
            if hp.train.checkpoint_dir is not None and (batch_id + 1) % 100 == 0:
                ae.eval()
                ckpt_model_filename = "ckpt_epoch_" + str(batch_id+1) + "_batch_id_" + str(batch_id+1) + ".pth"
                ckpt_model_path = os.path.join("new_checkpoints/", ckpt_model_filename)
                torch.save(ae.state_dict(), ckpt_model_path)
                ae.to(device).train()
        ae.eval()
        save_model_filename = "final_epoch_" + str(epoch + 1) + "_batch_id_" + str(batch_id + 1) + ".model"
        save_model_path = os.path.join(hp.train.checkpoint_dir, save_model_filename)
        torch.save(ae.state_dict(), save_model_path)

        print("\nDone, trained model saved at", save_model_path)


# In[14]:


#train()


# In[17]:


len(data_loader)


# In[ ]:





# In[ ]:





# In[18]:


ae = AutoEncoder().to('cuda')


# In[26]:


pytorch_total_params = sum(p.numel() for p in ae.parameters() if p.requires_grad)


# In[32]:


pytorch_total_params


# In[31]:


pytorch_total_params = sum(p.numel() for p in embedder_net.parameters() if p.requires_grad)


# In[33]:


dataset = TripletSpeakerDataset()
data_loader = torch.utils.data.DataLoader(dataset, drop_last=True)


# In[35]:


len(dataset)


# In[19]:


from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.manifold import TSNE
import time
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


# In[20]:


def preprocess_tsne(model, path):
    tsne = TSNE(verbose=1)
    utters = []
    files = glob.glob(path)
    speakers = []
    accents = []
    X1=[]
    for i,file in enumerate(files):
        if i%50!=0:
            continue
        try:
            pre_positive_speaker_embs, pre_positive_accent_embs = get_both_embs(file)
        except:
            print('continuing')
            continue
        speaker = file.split('/')[-2]
        accent = file.split('/')[-3]
        pre_positive_speaker_embs, pre_positive_accent_embs = Variable(torch.from_numpy(pre_positive_speaker_embs)).to('cpu'), Variable(torch.from_numpy(pre_positive_accent_embs)).to('cpu')
        utt, _ = ae.forward(pre_positive_speaker_embs, pre_positive_accent_embs)
        print(i)
        utters.append(utt)
        speakers.append(speaker)
        accents.append(accent)
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
    
    df2 = df.groupby(y).mean().reset_index()
    rndperm2 = np.random.permutation(df2.shape[0])
    """
    sns_plot = sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="y",
        palette=sns.color_palette("hls", 14),
        data=df.loc[rndperm,:],
        legend="full",
        alpha=1.0
    )
    sns_plot.figure.savefig('fig.png')
    """
    
    sns_plot2 = sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="index",
        palette=sns.color_palette("hls", 14),
        data=df2.loc[rndperm2,:],
        legend="full",
        alpha=1.0
    )
    sns_plot2.figure.savefig('fig2.png')
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
    """
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", 14),
        data=df_subset,
        legend="brief"
    )
    """
    return df


# In[21]:


def preprocess_tsne(model, path):
    tsne = TSNE(verbose=1)
    utters = []
    files = glob.glob(path)
    speakers = []
    accents = []
    X1=[]
    for i,file in enumerate(files):
        if i%50!=0:
            continue
        try:
            pre_positive_speaker_embs, pre_positive_accent_embs = get_both_embs(file)
        except:
            print('continuing')
            continue
        speaker = file.split('/')[-2]
        accent = file.split('/')[-3]
        pre_positive_speaker_embs, pre_positive_accent_embs = Variable(torch.from_numpy(pre_positive_speaker_embs)).to('cpu'), Variable(torch.from_numpy(pre_positive_accent_embs)).to('cpu')
        utt, _ = ae.forward(pre_positive_speaker_embs, pre_positive_accent_embs)
        print(i)
        utters.append(utt)
        speakers.append(speaker)
        accents.append(accent)
        X1.append(get_spectrogram_tisv(file))
    arr = [t.detach().numpy() for t in utters]
    y = np.asarray(accents)
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
    
    df2 = df.groupby(y).mean().reset_index()
    rndperm2 = np.random.permutation(df2.shape[0])
    """
    sns_plot = sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="y",
        palette=sns.color_palette("hls", 14),
        data=df.loc[rndperm,:],
        legend="full",
        alpha=1.0
    )
    sns_plot.figure.savefig('fig.png')
    """
    
    sns_plot2 = sns.lineplot(
        x="pca-one", y="pca-two",
        #hue="index",
        #palette=sns.color_palette("hls", 4),
        data=df2.loc[rndperm2,:],
        legend="full",
        alpha=1.0
    )
    sns_plot2 = sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="index",
        palette=sns.color_palette("hls", 4),
        data=df2.loc[rndperm2,:],
        legend="full",
        alpha=1.0
    )
    sns_plot2.figure.savefig('fig2.png')
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
    """
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", 14),
        data=df_subset,
        legend="brief"
    )
    """
    return df2


# In[ ]:




ae = AutoEncoder().to('cpu')
ae.load_state_dict(torch.load("./new_checkpoints/ckpt_epoch_5_batch_id_8000.pth")['state_dict'])
X, y ,specs= preprocess_tsne(ae, hp.data.train_path)


# In[25]:


len(y)


# In[26]:


df = tsne_visualize(X, y)


# In[27]:


df


# In[36]:


lens = []
for i in range(len(df)-1):
    dist = (df['pca-one'][i] - df['pca-one'][i+1])**2 + (df['pca-two'][i] - df['pca-two'][i+1])**2
    lens.append(np.sqrt(dist))


accs = ['American', 'Indian', 'Irish', 'Scottish']
for i in range(len(accs)):
    for j in range(i+1, len(accs)):
        rndperm2 = [i,j]
        print(accs[i], accs[j])
        plt.figure(figsize=(16,10))
        sns_plot2 = sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="index",
        palette=sns.color_palette("hls", 2),
        data=df.loc[rndperm2,:],
        legend="full",
        alpha=1.0
        )
        dist = (df['pca-one'][i] - df['pca-one'][j])**2 + (df['pca-two'][i] - df['pca-two'][j])**2
        sns_plot2.figure.savefig('{}{} - {}.png'.format(i, j, dist))


