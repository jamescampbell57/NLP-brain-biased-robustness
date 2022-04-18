from fmri_examples import *
import numpy as np
from scipy.io import loadmat

harry_potter = loadmat('subject_1.mat')

#5176
def assemble_harry_potter_dataset():
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

    dont_include_indices = [i for i in range(15)] + [i for i in range(335,355)] + [i for i in range(687,707)] + [i for i in range(966,986)] + [i for i in range(1346,1351)]

    X_fmri = harry_potter['data']

    useful_X_fmri = np.delete(X_fmri, dont_include_indices,axis=0)

    tr_times_arr = np.asarray(tr_times)

    useful_tr_times = np.delete(tr_times_arr, dont_include_indices)

    sentences = [[]]*1271
    for idx, useful_tr_time in enumerate(useful_tr_times):
        sentence= []
        for word, word_time in zip(words,word_times):
            if useful_tr_time - 10 <= word_time <= useful_tr_time:
                sentence.append(word)
        sentences[idx] = sentence

    return zip(sentences,useful_X_fmri)
