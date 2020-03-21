# prepare scratch directory
import os
import shutil
import tensorflow as tf
import matplotlib.pyplot as plt
import glob

SCRATCH_DIR = "scratch"

# utility function - not called from notebook
def load_slope_intercept(directory):
    slope_values = []
    intercept_values = []
    file_names = list(glob.glob(f'{directory}/weights_*.h5'))
    for file_name in sorted(file_names):
        epoch_model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    name="output",
                    units=1,
                    input_shape=(1,),
                    kernel_initializer=tf.keras.initializers.ones,
                    bias_initializer=tf.keras.initializers.zeros
                )
            ]
        )
        epoch_model.load_weights(file_name)
        weights = epoch_model.get_weights()
        slope = weights[0][0][0]
        intercept = weights[1][0]
        slope_values.append(slope)
        intercept_values.append(intercept)
    return slope_values, intercept_values

class Tracker:
    def __init__(self, name):
        # make scratch dirs
        self.dir = f"{SCRATCH_DIR}/{name}"
        os.makedirs(self.dir, exist_ok=True)
        for file_path in glob.glob(f'{self.dir}/weights_*.h5'):
            os.unlink(file_path)

    # used as a callback in fit(...)
    def get_callback(self):
        return tf.keras.callbacks.ModelCheckpoint(self.dir + '/weights_{epoch:02d}.h5', save_weights_only=True)

    def visualize_progress(self, **kwargs):
        m_true, b_true = kwargs["m_true"], kwargs["b_true"]
        sv, iv = load_slope_intercept(self.dir)
        plt.figure(figsize=(8,8))
        plt.scatter(sv, iv)
        for idx, point in enumerate(zip(sv, iv)):
            plt.annotate(str(idx), point)

        # show the correct parameter values as a reference point
        plt.scatter(m_true, b_true, c="r")
        plt.annotate("correct", (m_true, b_true))

        plt.xlabel("slope")
        plt.ylabel("intercept")
        plt.show()
