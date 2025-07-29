# import standard libraries
import cv2
import os
import random
import numpy as np

# import tensorflow functions
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf




def main():
    # limit gpu growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # initialize folders, positive negative and anchor
    pos_path = os.path.join('data','positive')
    neg_path = os.path.join('data','negative')
    anc_path = os.path.join('data','anchor')

    # make the directories
    if not os.path.exists(pos_path) and not os.path.exists(neg_path) and not os.path.exists(anc_path):
        os.makedirs(pos_path)
        os.makedirs(neg_path)
        os.makedirs(anc_path)
    else:
        print("the files have already been made")


if __name__ == '__main__':
    main()