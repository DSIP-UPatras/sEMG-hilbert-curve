# Evaluation tools
# https://github.com/GeekLiB/keras/blob/master/keras/metrics.py
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import top_k_categorical_accuracy
import numpy as np
from sklearn import metrics


def top_1_accuracy(y_true, y_pred):
    """ Calculates top-1 accuracy of the predictions. To be used as evaluation metric in model.compile().

    Arguments:
        y_true -- array-like, true labels
        y_pred -- array-like, predicted labels

    Returns:
        top-1 accuracy
    """
    return top_k_categorical_accuracy(y_true, y_pred, k=1)


def top_3_accuracy(y_true, y_pred):
    """ Calculates top-3 accuracy of the predictions. To be used as evaluation metric in model.compile().

    Arguments:
        y_true -- array-like, true labels
        y_pred -- array-like, predicted labels

    Returns:
        top-3 accuracy
    """
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


def top_5_accuracy(y_true, y_pred):
    """ Calculates top-5 accuracy of the predictions. To be used as evaluation metric in model.compile().

    Arguments:
        y_true -- array-like, true labels
        y_pred -- array-like, predicted labels

    Returns:
        top-5 accuracy
    """
    return top_k_categorical_accuracy(y_true, y_pred, k=5)


def kullback_leibler_divergence(y_true, y_pred):
    '''Calculates the Kullback-Leibler (KL) divergence between prediction
    and target values.
    '''
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.sum(y_true * K.log(y_true / y_pred), axis=-1)


def poisson(y_true, y_pred):
    '''Calculates the poisson function over prediction and target values.
    '''
    return K.mean(y_pred - y_true * K.log(y_pred + K.epsilon()))


def cosine_proximity(y_true, y_pred):
    '''Calculates the cosine similarity between the prediction and target
    values.
    '''
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return -K.mean(y_true * y_pred)


def matthews_correlation(y_true, y_pred):
    '''Calculates the Matthews correlation coefficient measure for quality
    of binary classification problems.
    '''
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())


def precision(y_true, y_pred):
    '''Calculates the precision, a metric for multi-label classification of
    how many selected items are relevant.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    '''Calculates the recall, a metric for multi-label classification of
    how many relevant items are selected.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    '''Calculates the F score, the weighted harmonic mean of precision and recall.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    '''
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')
        
    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    '''Calculates the f-measure, the harmonic mean of precision and recall.
    '''
    return fbeta_score(y_true, y_pred, beta=1)


def evaluate_vote(y_true, y_pred, reps):
    """ Evaluates model based on majority voting of a complete repetition segment.

    Arguments:
        y_true -- array-like, true class labels of EMG images
        y_pred -- array-like, predicted class labels of EMG images
        reps -- array-like, repetition of each sample
        group -- array-like, similar to y_true, but rest labels '0' are assigned to the movement that follows
        vote_window -- integer, size in frames of majority voting window. The increment of voting windows is 1 frame.

    Returns:
        accuracy_vote -- float, accuracy metric for majority voting
        cnf_matrix_vote -- array-like, confusion matrix
    """
    # Vote
    y_true_vote = []
    y_pred_vote = []
    max_label = np.max(y_true)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    reps = np.array(reps)
    assert (y_true.size == reps.size), 'Error (y_true.size = {}) != (reps.size = {})'.format(y_true.size, reps.size)
    for m in np.unique(y_true):
        im = np.isin(y_true, m)
        for r in np.unique(reps):
            ir = np.isin(reps, r)

            # For movement
            irm = np.logical_and(im, ir)
            y_test_rm = y_true[irm].astype(int)
            y_pred_rm = y_pred[irm].astype(int)

            if y_pred_rm.size > 0:
                bins = np.bincount(y_pred_rm, minlength=max_label + 1)
                k = np.argmax(bins)
                # print('({},{}): {}, {}'.format(m,r,bins,k))
                y_true_vote.append(m)
                y_pred_vote.append(k)

    # Vote accuracy
    cnf_matrix_vote = metrics.confusion_matrix(
        y_true_vote, y_pred_vote, labels=np.unique(y_true))
    accuracy_vote = metrics.accuracy_score(y_true_vote, y_pred_vote)
    # print(cnf_matrix_vote)
    # print(accuracy_vote)
    return accuracy_vote, cnf_matrix_vote


def confidence_threshold_categorical_loss(threshold):
    

    def compute_loss(y_true, y_pred):

        pred_label_probs = K.reduce_sum(y_true * y_pred)    # predicted probs of correct labels
        loss_mask = pred_label_probs < threshold            # compute loss for less confident preds
        loss = K.categorical_crossentropy(y_true[loss_mask], y_pred[loss_mask])
        return loss

    return compute_loss


