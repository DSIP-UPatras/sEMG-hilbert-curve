# Optimization tools
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import TensorBoard, Callback

class MyTensorboard(TensorBoard):
    """ Tensorboard callback to store the learning rate at the end of each epoch.
    """
    def __init__(self, **kwargs):
        kwargs['histogram_freq'] = 0
        kwargs['write_graph'] = False
        kwargs['write_grads'] = False
        kwargs['write_images'] = False
        kwargs['embeddings_freq'] = 0
        #kwargs['update_freq'] = 'epoch'
        super(MyTensorboard, self).__init__(**kwargs)

    def on_epoch_end(self, epoch, logs=None):
        lr = K.eval(self.model.optimizer.lr)
        lr_summary = tf.Summary(
            value=[tf.Summary.Value(tag='lr', simple_value=lr)])
        self.writer.add_summary(lr_summary, epoch)
        self.writer.flush()
        super(MyTensorboard, self).on_epoch_end(epoch, logs)


class MyLRScheduler(Callback):
    def __init__(self, schedule_type = 'constant', decay = 0, step = 1, step_epochs = 0, max_epochs = 100, lr_start = 0, lr_end = 0, verbose=0):
        super(MyLRScheduler, self).__init__()
        self.schedule_type = schedule_type
        self.decay = float(decay)
        self.step = step
        self.max_epochs = max_epochs
        if step_epochs == 0:
            self.step_epochs = np.arange(self.step, self.max_epochs, self.step)
        else:
            self.step_epochs = list(step_epochs)
        self.lr_start = float(lr_start)
        self.lr_end = float(lr_end)
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = self.schedule(epoch)
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nEpoch %05d: LearningRateScheduler reducing learning '
                  'rate to %s.' % (epoch + 1, lr))

    def schedule(self, epoch):
        """ Defines the learning rate schedule. This is called at the begin of each epoch through the LearningRateScheduler callback.
            Arguments:
                epoch -- integer, current epoch, [0, #epochs-1]

            Returns:
                rate -- calculated learning rate
        """
        if self.schedule_type == 'constant':
            rate = self.lr_start
        elif self.schedule_type == 'decay' or self.schedule_type == 'step':
            i = np.searchsorted(self.step_epochs, epoch, side='right')
            rate = self.lr_start * (self.decay ** i)
        # elif self.schedule_type == 'step':
        #     rate = self.lr_start * (self.decay ** np.floor(epoch / self.step))
        elif self.schedule_type == 'anneal':
            rate = self.lr_start / (1 + self.decay * epoch)
        elif self.schedule_type == 'clr_triangular':
            e = epoch + self.step
            c = np.floor(1 + e / (2 * self.step))
            x = np.abs(e / self.step - 2 * c + 1)
            rate = self.lr_end + (self.lr_start - self.lr_end) * \
                np.maximum(0, (1 - x)) * float(self.decay**(c - 1))
        elif self.schedule_type == 'clr_restarts':
            c = np.floor(epoch / self.step)
            x = 1 + np.cos((epoch % self.step) / self.step * np.pi)
            rate = self.lr_end + 0.5 * (self.lr_start - self.lr_end) * x * self.decay**c
        elif self.schedule_type == 'warmup':
            # rate = self.lr_start * np.min(np.pow(epoch, -0.5), epoch / self.step)
            if epoch <= self.step:
                rate = self.lr_start * epoch / self.step
            else:
                rate = self.lr_start * (self.decay ** (epoch - self.step))
        else:
            raise ValueError('Not supported learning schedule.')
        return float(rate)


