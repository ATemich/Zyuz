import os
import numpy as np
import tensorflow as tf
from .fr_utils import *
from .inception_blocks_v2 import *
from .. import FaceRecognizer
from keras import backend as K
K.set_image_data_format('channels_first')

class FaceNet(FaceRecognizer):
    def __init__(self):
        super().__init__()
        self.database = {}
        self.FRmodel = faceRecoModel(input_shape=(3, 96, 96))
        self.FRmodel.compile(optimizer='adam', loss=self.triplet_loss, metrics=['accuracy'])
        load_weights_from_FaceNet(self.FRmodel)

    def load_from_images(self, path):
        for file in os.listdir(path):
            file = os.path.join(path, file)
            identity = os.path.splitext(os.path.basename(file))[0]
            self.database[identity] = img_path_to_encoding(file, self.FRmodel)

    def recognize(self, pic, face=None):
        image = self.cut_out(pic, face)
        encoding = img_to_encoding(image, self.FRmodel)

        min_dist = float('inf')
        identity = None

        for name, db_enc in self.database.items():
            dist = np.linalg.norm(db_enc - encoding)
            if dist < min_dist:
                min_dist = dist
                identity = name

        if min_dist > 0.52:
            return None
        else:
            return str(identity)

    @staticmethod
    def triplet_loss(y_true, y_pred, alpha=0.3):
        """
        Implementation of the triplet loss as defined by formula (3)

        Arguments:
        y_pred -- python list containing three objects:
                anchor -- the encodings for the anchor images, of shape (None, 128)
                positive -- the encodings for the positive images, of shape (None, 128)
                negative -- the encodings for the negative images, of shape (None, 128)

        Returns:
        loss -- real number, value of the loss
        """

        anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

        # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
        # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
        # Step 3: subtract the two previous distances and add alpha.
        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
        # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
        loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))

        return loss
