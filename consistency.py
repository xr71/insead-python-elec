# Do not pay attention to this file - it is imported and used in the lessons
# but you do not need to worry about it
# If you are curious, read:
# https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
import numpy as np
import tensorflow as tf
import random as rn

def remove_randomness():
    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.

    np.random.seed(42)

    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.

    rn.seed(12345)

    # Force TensorFlow to use single thread.
    # Multiple threads are a potential source of non-reproducible results.
    # For further details, see: https://stackoverflow.com/questions/42022950/

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    # The below tf.random.set_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    tf.random.set_seed(1234)
