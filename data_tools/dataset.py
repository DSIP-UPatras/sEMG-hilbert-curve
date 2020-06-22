import numpy as np
import scipy.io
import os.path
import glob


def load_dataset(input_directory, subject, classes, repetitions, rest_reps=None):
    '''Loads data and applies preprocess_function_1'''
    X, y, r = [], [], []
    if rest_reps is None:
        rest_reps = len(repetitions)
    n_reps = len(repetitions)
    n_classes = len(classes)
    subject = subject if isinstance(subject, list) else [subject]

    for d in subject:
        for label in classes:
            if label == 0:
                rest_rep_groups = list(
                    zip(
                        np.random.choice(repetitions, (rest_reps), replace=rest_reps > n_reps),
                        np.random.choice([i for i in classes if i != 0], (rest_reps), replace=rest_reps > n_classes-1)
                        )
                    )
                for rep, lab in rest_rep_groups:
                    file = '{}/subject-{:02d}/gesture-00/rep-{:02d}_{:02d}.mat'.format(input_directory, d, int(rep), int(lab))
                    if os.path.isfile(file):
                        data = scipy.io.loadmat(file)
                        x = data['emg'].copy()

                        X.append(x)
                        y.append(int(np.squeeze(data['stimulus'])[0]))
                        r.append(int(np.squeeze(data['repetition'])[0]))
            else:
                for rep in repetitions:
                    file = '{}/subject-{:02d}/gesture-{:02d}/rep-{:02d}.mat'.format(input_directory, d, int(label), int(rep))
                    if os.path.isfile(file):
                        data = scipy.io.loadmat(file)
                        x = data['emg'].copy()

                        X.append(x)
                        y.append(int(np.squeeze(data['stimulus'])[0]))
                        r.append(int(np.squeeze(data['repetition'])[0]))

    return X, y, r

def make_segments(data, window_size, window_step):
        '''Creates segments either using predefined step'''
        x_offsets = []

        if window_size != 0:
            if (window_step is not None) and (window_step > 0):
                for i in range(len(data)):
                    for j in range(0, len(data[i]) - window_size, window_step):
                        x_offsets.append((i, j))
            else:
                for i in range(len(data)):
                    x_offsets.append((i, max(0,len(data[i])//2 - window_size//2)))
        else:
            x_offsets = [(i, 0) for i in range(len(data))]

        return x_offsets
    
def get_window(sig, window_size, index):
    window_size = min(window_size, len(sig))
    return np.copy(sig[index:index+window_size])
