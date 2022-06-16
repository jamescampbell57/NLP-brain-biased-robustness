import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import BertModel
import numpy as np
from scipy.io import loadmat
import os


class HarryPotter(Dataset):
    def __init__(self, subject):

    data_path = '/home/ubuntu/nlp-brain-biased-robustness/data/harry_potter_brain'
    # install dataset
    if not os.path.exists(data_path):
        os.system('mkdir '+data_path)
        for i in range(1,9):
            os.system(f'wget http://www.cs.cmu.edu/~fmri/plosone/files/subject_{i}.mat -P '+data_path)

    harry_potter = loadmat(os.paths.join(data_path, f'subject_{subject}.mat'))

    words = []
    for i in range(5176):
        word = harry_potter['words'][0][i][0][0][0][0]
        words.append(word)

    word_times = []
    for i in range(5176):
        word_time = harry_potter['words'][0][i][1][0][0]
        word_times.append(word_time)

    tr_times = []
    for i in range(1351):
        tr_time = harry_potter['time'][i,0]
        tr_times.append(tr_time)

    #dont_include_indices = []
    #for idx, tr_time in enumerate(tr_times):
    #    if not set(np.arange(tr_time - 10, tr_time, .5)).issubset(set(word_times)):
    #        dont_include_indices.append(idx)

    dont_include_indices = [i for i in range(15)] + [i for i in range(335,355)] + [i for i in range(687,707)] + [i for i in range(966,986)] + [i for i in range(1346,1351)]

    X_fmri = harry_potter['data']

    useful_X_fmri = np.delete(X_fmri, dont_include_indices,axis=0)

    #tr_times_arr = np.asarray(tr_times)

    useful_tr_times = np.delete(np.asarray(tr_times), dont_include_indices)

    sentences = [[]]*1271
    for idx, useful_tr_time in enumerate(useful_tr_times):
        sentence= []
        for word, word_time in zip(words,word_times):
            if useful_tr_time - 10 <= word_time <= useful_tr_time:
                sentence.append(word)
        sentences[idx] = sentence   

    actual_sentences = ['']*1271
    for idx, sentence in enumerate(sentences):
        for word in sentence:
            actual_sentences[idx] = actual_sentences[idx] + word + ' '


    fmri = torch.as_tensor(useful_X_fmri)
    truth_fmri = fmri[:5,:]



def __getitem__(self, idx):
    return self.fmri_data[idx]
    
def __len__(self):
    return len(self.fmri_data)

