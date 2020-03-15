#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 20:55:52 2018

@author: harry
"""
import glob
import numpy as np
import os
import random
from random import shuffle
import torch
from torch.utils.data import Dataset
from hparam import speakers as spa
from hparam import hparam as hp
from utils import mfccs_and_spec

import librosa
class SpeakerDataset(Dataset):

    def __init__(self, shuffle=True, utter_start=0):

        # data path
        if hp.training:
            self.path = hp.data.train_path
            self.utter_num = hp.train.M
        else:
            self.path = hp.data.test_path
            self.utter_num = hp.test.M
        self.file_list = os.listdir(self.path)
        self.shuffle=shuffle
        self.utter_start = utter_start

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        np_file_list = os.listdir(self.path)

        positive = random.sample(np_file_list, 1)[0]  # select random speaker
        negative = random.sample(np_file_list, 1)[0]  # select random speaker

        positive_speaker = int(positive.split('_')[0].split('speaker')[1])
        negative_speaker = int(negative.split('_')[0].split('speaker')[1])

        while positive_speaker == negative_speaker:
            negative = random.sample(np_file_list, 1)[0]
            negative_speaker = int(negative.split('_')[0].split('speaker')[1])


        positive_id = int(positive.split('_')[1].split('.')[0])
        negative_id = int(negative.split('_')[1].split('.')[0])

        anchor = random.sample(np_file_list, 1)[0]
        anchor_speaker = int(anchor.split('_')[0].split('speaker')[1])
        anchor_id = int(negative.split('_')[1].split('.')[0])
        while anchor_speaker!=positive_speaker and anchor_id == positive_id:
            anchor = random.sample(np_file_list, 1)[0]
            anchor_speaker = int(anchor.split('_')[0].split('speaker')[1])
            anchor_id = int(negative.split('_')[1].split('.')[0])


        positive_utters = np.load(os.path.join(self.path, positive))        # load utterance spectrogram of selected speaker
        negative_utters = np.load(os.path.join(self.path, negative))        # load utterance spectrogram of selected speaker
        anchor_utters = np.load(os.path.join(self.path, anchor))

        positive_utters = torch.tensor(np.transpose(positive_utters, axes=(0,2,1)))     # transpose [batch, frames, n_mels]
        negative_utters = torch.tensor(np.transpose(negative_utters, axes=(0,2,1)))
        anchor_utters = torch.tensor(np.transpose(anchor_utters, axes=(0,2,1)))

        return anchor_utters, positive_utters, negative_utters, positive_id, negative_id


class TripletSpeakerDataset(Dataset):

    def __init__(self, shuffle=True, utter_start=0):

        # data path
        if hp.training:
            self.path = hp.data.train_path
            self.utter_num = hp.train.M
        else:
            self.path = hp.data.test_path
            self.utter_num = hp.test.M
        self.file_list = glob.glob(self.path)
        self.shuffle=shuffle
        self.utter_start = utter_start

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        #np_file_list = os.listdir(self.path)

        accents = ["Scottish", "Irish", "American", "Indian"]
        n_data_scot=0
        n_data_irish=0
        n_data_amer=0
        n_data_ind=0

        data_scot = []
        data_irish = []
        data_amer = []
        data_ind = []


        data_path = "data/"

        positive_accent = random.sample(accents, 1)[0]
        negative_accent = random.sample(accents, 1)[0]
        while positive_accent == negative_accent:
            negative_accent = random.sample(accents, 1)[0]

        accent_positive_speakers = os.listdir('data/' + positive_accent)
        accent_negative_speakers = os.listdir('data/' + negative_accent)

        positive_speaker = random.sample(accent_positive_speakers, 1)[0]
        negative_speaker_accent = random.sample(accent_negative_speakers, 1)[0]

        accent_negative_utter = random.sample(os.listdir('data/'+negative_accent+'/'+negative_speaker_accent), 1)[0]

        negative_speaker = random.sample(accent_positive_speakers, 1)[0]

        while positive_speaker == negative_speaker:
            negative_speaker = random.sample(accent_positive_speakers, 1)[0]

        accent_negative_utter = random.sample(os.listdir('data/' + negative_accent+'/'+negative_speaker_accent), 1)[0]
        accent_negative_path = 'data/'+negative_accent+'/'+negative_speaker_accent + '/'+accent_negative_utter


        positive_files = os.listdir('data/'+positive_accent+'/'+positive_speaker)
        negative_files = os.listdir('data/'+positive_accent+'/'+negative_speaker)


        positive = random.sample(positive_files, 1)[0]
        anchor = random.sample(positive_files, 1)[0]
        negative = random.sample(negative_files, 1)[0]


        while positive==anchor:
            anchor = random.sample(positive_files, 1)[0]

        utterance_positive_id = int(positive.split('_')[1].split('.')[0])
        utterance_negative_id = int(negative.split('_')[1].split('.')[0])
        utterance_anchor_id = int(anchor.split('_')[1].split('.')[0])

        speaker_positive_id = positive.split('_')[0].split('.')[0]
        speaker_negative_id = negative.split('_')[0].split('.')[0]
        speaker_anchor_id = anchor.split('_')[0].split('.')[0]


        positive_utter_path = 'data/'+positive_accent+'/'+positive_speaker+'/'+positive
        anchor_utter_path = 'data/'+positive_accent+'/'+positive_speaker+'/'+anchor
        negative_utter_path = 'data/'+positive_accent+'/'+negative_speaker+'/'+negative
        accent_negative_path = 'data/'+negative_accent+'/'+negative_speaker_accent + '/'+accent_negative_utter


        """
        while anchor_speaker!=positive_speaker and anchor_id == positive_id:
            anchor = random.sample(np_file_list, 1)[0]
            anchor_speaker = int(anchor.split('_')[0].split('speaker')[1])
            anchor_id = int(negative.split('_')[1].split('.')[0])
        """

        #positive_utters, _ = librosa.load(positive_utter_path)        # load utterance spectrogram of selected speaker
        #negative_utters, _ = librosa.load(negative_utter_path)        # load utterance spectrogram of selected speaker
        #anchor_utters, _ = librosa.load(anchor_utter_path)
        #negative_accent_utter, _ = librosa.load(accent_negative_path)

        return anchor_utter_path, positive_utter_path, negative_utter_path, accent_negative_path, speaker_positive_id, speaker_negative_id, speaker_anchor_id, positive_accent, negative_accent
