# Run experiment
import sys
import os
# set environment variable as `docker run -e CUDA_VISIBLE_DEVICES=0 <image-name>`
# GPU_ID = 0 
# os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

import rng_init
import getopt
import json
import copy
import scipy
import tensorflow.keras as keras
import tensorflow as tf
import sklearn
import numpy as np
from  data_tools import dataset
from data_tools import pipelines
from model_tools import evaluation, optimization
from sklearn.pipeline import Pipeline
from models import *
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


print('Keras:', keras.__version__)
print('Tensorflow:', tf.__version__)

long_options = [
    "subject=", "model=", "timestamp=", "log=", "rng=",
    "img_height=", "img_width=", "img_depth=",
    "window_size=", "window_step=",
    "augment_factor=", "augment_jitter=", "augment_mwrp=",
    "hilbert_type=",
    "model_params=", "dropout=",
    "epochs=", "batch_size=",
    "validation", "include_rest_gesture"
]

try:
    opts, args = getopt.getopt(sys.argv[1:], "", long_options)
    print(opts)
except getopt.GetoptError:
    print('run_experiment_hilberty.py {}'.format(long_options))
    sys.exit(2)

SUBJECT = MODEL = TIMESTAMP = LOG = RNG = \
    D_HEIGHT = D_WIDTH = D_DEPTH = D_WINDOW_SIZE = D_WINDOW_STEP = \
    D_AUGMENT_FACTOR = D_JITTER = D_MWRP = D_HILBERT_MAP = \
    N_MODEL_PARAMS = N_DROPOUT = O_EPOCHS = O_BATCH = None
VALIDATION = False
D_INCLUDE_REST = False

for opt, arg in opts:
    if opt == "--subject":
        SUBJECT = int(arg)
    elif opt == "--model":
        MODEL = str(arg)
    elif opt == "--timestamp":
        TIMESTAMP = int(arg)
    elif opt == "--log":
        LOG = str(arg)
    elif opt == "--rng":
        RNG = int(arg)
    elif opt == "--img_height":
        D_HEIGHT = int(arg)     # int(np.power(2,np.ceil(np.log2(np.sqrt(D_WINDOW)))))
    elif opt == "--img_width":
        D_WIDTH = int(arg)
    elif opt == "--img_depth":
        D_DEPTH = int(arg)
    elif opt == "--window_size":
        D_WINDOW_SIZE = int(arg)
    elif opt == "--window_step":
        D_WINDOW_STEP = int(arg)
    elif opt == "--augment_factor":
        D_AUGMENT_FACTOR = int(arg)
    elif opt == "--augment_jitter":
        D_JITTER = int(arg)
    elif opt == "--augment_mwrp":
        D_MWRP = float(arg)
    elif opt == "--hilbert_type":
        D_HILBERT_MAP = str(arg)            # (none/None, time, electrodes)
    elif opt == "--model_params":
        N_MODEL_PARAMS = str(arg)
    elif opt == "--dropout":
        N_DROPOUT = float(arg)
    elif opt == "--epochs":
        O_EPOCHS = int(arg)
    elif opt == "--batch_size":
        O_BATCH = int(arg)
    elif opt == "--validation":
        VALIDATION = True
    elif opt == "--include_rest_gesture":
        D_INCLUDE_REST = True


if MODEL == 'VGG':
    model_fun = VGGNet
elif MODEL == 'DENSE':
    model_fun = DenseNet
elif MODEL == 'SQUEEZE':
    model_fun = SqueezeNet
elif MODEL == 'MSHILB':
    model_fun = MSHilbNet
else:
    raise ValueError('Unsupported model, {}'.format(MODEL))

D_INPUT_DIRECTORY = os.path.abspath('Datasets/Datasets/Ninapro-DB1')
if D_INCLUDE_REST:
    D_GESTURES = [i for i in range(0, 53)]
else:
    D_GESTURES = [i for i in range(1, 53)]
if VALIDATION:
    D_TRAIN_REPS = [1,3,4,8,9,10]
    D_TEST_REPS = [6]
else:
    D_TRAIN_REPS = [1, 3, 4, 6, 8, 9, 10]
    D_TEST_REPS = [2, 5, 7]

D_NORMALIZE = False

with open(os.path.abspath('configs')+os.sep+N_MODEL_PARAMS) as json_file:
    model_params = json.loads(json_file.read())
if N_DROPOUT:
    model_params['n_dropout'] = N_DROPOUT
N_OUTPUT_NAME = model_params['output_layer_name']


O_LOSS = 'categorical_crossentropy'
O_METRICS = [
    evaluation.top_1_accuracy, evaluation.top_3_accuracy, evaluation.top_5_accuracy,
    evaluation.precision, evaluation.recall
]
O_OPTIMIZER = 'sgd'
O_LRATE = {'schedule_type': 'step', 'decay': 0.5, 'step': 10, 'lr_start': 0.1}  # model selection
O_LRATE = {'schedule_type': 'step', 'decay': 0.5, 'step': 15, 'lr_start': 0.1}  # normal


LOGGING_FILE_PREFIX = LOG + '_' + str(TIMESTAMP)
LOGGING_TENSORBOARD_FILE = os.path.abspath('logs') + os.sep + 'tblogs' + os.sep + 'L_' + LOGGING_FILE_PREFIX
LOGGING_FILE = os.path.abspath('logs') + os.sep + 'L_' + LOGGING_FILE_PREFIX + '.csv'
METRICS_SAVE_FILE = os.path.abspath('metrics') + os.sep + 'M_' + LOGGING_FILE_PREFIX + '_{}.mat'.format(SUBJECT)
MODEL_SAVE_FILE = os.path.abspath('models') + os.sep + 'M_' + LOGGING_FILE_PREFIX + '_{}.h5'.format(SUBJECT)
MODEL_BEST_SAVE_FILE = os.path.abspath('models') + os.sep + 'M_' + LOGGING_FILE_PREFIX + '_{}_BEST.h5'.format(SUBJECT)

if os.path.isfile(LOGGING_FILE) is False:
    with open(LOGGING_FILE, 'w') as f:
        f.write(
            '{}\n{}\n'.format(opts, args)
        )

evals = {}

print('Subject: {}'.format(SUBJECT))
print('LOG file: {}'.format(LOGGING_FILE_PREFIX))

#############################################################################
#############################################################################
# LOAD DATASET
#############################################################################
X_train, Y_train, _ = dataset.load_dataset(
    D_INPUT_DIRECTORY, SUBJECT, D_GESTURES, D_TRAIN_REPS
)
X_test, Y_test, r_test = dataset.load_dataset(
    D_INPUT_DIRECTORY, SUBJECT, D_GESTURES, D_TEST_REPS
)

le = sklearn.preprocessing.LabelEncoder()
le.fit(Y_train)
Y_train = le.transform(Y_train).tolist()
Y_test = le.transform(Y_test).tolist()


#############################################################################
#############################################################################
# PREPROCESS
#############################################################################
# 1. LPF/NORM
preproc = Pipeline([
    ('lowpass', pipelines.preprocessing.TrfLowpass(f=1, fs=100, order=1)),
    ('normalize', pipelines.preprocessing.TrfNormalize())
])

preproc.fit(X_train, Y_train)
X_train = preproc.transform(X_train)
X_test = preproc.transform(X_test)

# 2. AUGMENT
x_temp = []
augment = Pipeline([
    ('mwarp', pipelines.augmentation.TrfMagWarp(D_MWRP)),
    ('jitter', pipelines.augmentation.TrfJitter(D_JITTER))
])

for _ in range(D_AUGMENT_FACTOR):
    x_temp.append(augment.transform(X_train))

print(f"Merge initial train and augmented")
X_train = np.concatenate([X_train, np.concatenate(x_temp)])
Y_train = np.concatenate([Y_train, np.tile(Y_train, D_AUGMENT_FACTOR)])

print('len(X_train): {}\nlen(Y_train): {}'.format(len(X_train), len(Y_train)))

# 3. WINDOW
if D_WINDOW_SIZE is not None:
    offsets_train = dataset.make_segments(X_train, D_WINDOW_SIZE, D_WINDOW_STEP)    # overlapped windows for training
    offsets_test = dataset.make_segments(X_test, D_WINDOW_SIZE, D_WINDOW_SIZE)      # no overlap for testing
    X_train_segments, Y_train_segments = [], []
    X_test_segments, Y_test_segments, r_test_segments = [], [], []
    for offset in offsets_train:
        i, j = offset
        X_train_segments.append(dataset.get_window(X_train[i], D_WINDOW_SIZE, j))
        Y_train_segments.append(Y_train[i])
    for offset in offsets_test:
        i, j = offset
        X_test_segments.append(dataset.get_window(X_test[i], D_WINDOW_SIZE, j))
        Y_test_segments.append(Y_test[i])
        r_test_segments.append(r_test[i])
else:
    X_train_segments, Y_train_segments = X_train, Y_train
    X_test_segments, Y_test_segments, r_test_segments = X_test, Y_test, r_test

# 4. HILBERT CURVE
if D_HILBERT_MAP not in ("none", "None"):
    axis = 0 if 'time' in D_HILBERT_MAP else 1
    hilbert = Pipeline([
        ('hilbert', pipelines.transformations.TrfSpaceFillCurve('HilbertCurve', axis))
    ])

# 5. LABELS TO CATEGORICAL
Y_train_segments = keras.utils.to_categorical(Y_train_segments, num_classes=len(D_GESTURES))
Y_test_segments = keras.utils.to_categorical(Y_test_segments, num_classes=len(D_GESTURES))

# 6. RESHAPE
X_train_segments = np.array(X_train_segments)
Y_train_segments = np.array(Y_train_segments)
X_test_segments = np.array(X_test_segments)
Y_test_segments = np.array(Y_test_segments)
N_INPUT_SHAPE = X_train_segments[0].shape

print('shape(X_train_segments): {}\nshape(Y_train_segments): {}'.format(X_train_segments.shape, Y_train_segments.shape))
print('shape(X_test_segments): {}\nshape(Y_test_segments): {}'.format(X_test_segments.shape, Y_test_segments.shape))

# 7. SHUFFLE
X_train_segments, Y_train_segments = sklearn.utils.shuffle(X_train_segments, Y_train_segments)

#############################################################################
#############################################################################
# DEFINE MODEL
#############################################################################

model = model_fun(N_INPUT_SHAPE, len(D_GESTURES), **model_params)
if O_OPTIMIZER == 'sgd':
    opt = keras.optimizers.SGD(lr=O_LRATE['lr_start'], momentum=0.9)
elif O_OPTIMIZER == 'adam':
    opt = keras.optimizers.Adam(lr=O_LRATE['lr_start'])
model.compile(
    optimizer=opt,
    loss=O_LOSS,
    metrics=O_METRICS
)
model.summary()

train_callbacks = []
tensorboardCallback = optimization.MyTensorboard(
    log_dir=LOGGING_TENSORBOARD_FILE + "/{}".format(SUBJECT),
    batch_size=100,
    histogram_freq=O_EPOCHS // 5
)
train_callbacks.append(tensorboardCallback)
lrScheduler = optimization.MyLRScheduler(**O_LRATE)
train_callbacks.append(lrScheduler)
mdlCheckpoint = ModelCheckpoint(MODEL_BEST_SAVE_FILE, monitor='val_loss', save_best_only=True)
train_callbacks.append(mdlCheckpoint)

#############################################################################
#############################################################################
# TRAIN
#############################################################################
history = model.fit(
    x=X_train_segments, y={N_OUTPUT_NAME: Y_train_segments},
    epochs=O_EPOCHS, batch_size=O_BATCH,
    shuffle=True,
    validation_data=(X_test_segments, {N_OUTPUT_NAME: Y_test_segments}),
    callbacks=train_callbacks
)
#############################################################################

#############################################################################
#############################################################################
# SAVE MODEL
#############################################################################
model.save(MODEL_SAVE_FILE)
print("Saved model to disk")

#############################################################################
#############################################################################
# EVALUATE
#############################################################################
Y_pred = model.predict(X_test_segments)
y_test_pred = np.argmax(Y_pred, axis=1)
y_test = np.argmax(Y_test_segments, axis=1)

# train_accuracy = sklearn.metrics.accuracy_score(y_train, y_train_pred)
test_accuracy = sklearn.metrics.accuracy_score(y_test, y_test_pred)
test_precision = sklearn.metrics.precision_score(y_test, y_test_pred, average='weighted')
test_recall = sklearn.metrics.recall_score(y_test, y_test_pred, average='weighted')
test_f1 = sklearn.metrics.f1_score(y_test, y_test_pred, average='weighted')

# CM
# C_{i, j} is equal to the number of observations known to be in group i but predicted to be in group j.
cnf_matrix_frame = sklearn.metrics.confusion_matrix(y_test, y_test_pred)

# VOTE
accuracy_vote, cnf_matrix_vote = evaluation.evaluate_vote(y_test, y_test_pred, r_test_segments)

evals['train_top_1_acc'] = history.history['top_1_accuracy']
evals['test_top_1_acc'] = history.history['val_top_1_accuracy']
evals['train_loss'] = history.history['loss']
evals['test_loss'] = history.history['val_loss']
evals['test_top_3_acc'] = history.history['val_top_3_accuracy']
evals['test_top_5_acc'] = history.history['val_top_5_accuracy']
evals['test_cm'] = cnf_matrix_frame
evals['test_cm_vote'] = cnf_matrix_vote
evals['test_vote_acc'] = accuracy_vote
evals['test_acc'] = test_accuracy
evals['test_pr'] = test_precision
evals['test_re'] = test_recall
evals['test_f1'] = test_f1

with open(LOGGING_FILE, 'a') as f:
    f.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(
        SUBJECT, N_INPUT_SHAPE, len(X_train_segments), len(X_test_segments),
        evals['train_loss'], evals['train_top_1_acc'],
        evals['test_loss'], evals['test_top_1_acc'], evals['test_vote_acc'], evals['test_top_3_acc'], evals['test_top_5_acc'],
        evals['test_acc'], evals['test_pr'], evals['test_re'], evals['test_f1']
    ))

scipy.io.savemat(METRICS_SAVE_FILE.format(SUBJECT), evals)
