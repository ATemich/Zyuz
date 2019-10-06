import os
import cv2
import numpy as np
import tensorflow as tf
from .inception_blocks_v2 import faceRecoModel
from .. import FaceRecognizer
from keras import backend as K
K.set_image_data_format('channels_first')

class FaceNet(FaceRecognizer):
    def __init__(self, threshold=0.53):
        super().__init__()
        self.threshold = threshold
        self.database = {}
        self.model = faceRecoModel(input_shape=(3, 96, 96))
        self.model.compile(optimizer='adam', loss=self.triplet_loss, metrics=['accuracy'])

        path = os.path.dirname(os.path.realpath(__file__))
        self.model.load_weights(os.path.join(path, 'weights.h5'))

    def load_from_images(self, path):
        for file in os.listdir(path):
            if file[0] != '.':
                file = os.path.join(path, file)
                id = os.path.splitext(os.path.basename(file))[0].split()[0]
                self.database[id] = self.database.get(id, [])+[self.img_path_to_encoding(file)]

    def recognize_image(self, image):
        encoding = self.img_to_encoding(image)

        min_dist = float('inf')
        identity = None

        for name, db_encs in self.database.items():
            for db_enc in db_encs:
                dist = np.linalg.norm(db_enc - encoding)
                if dist < min_dist:
                    min_dist = dist
                    identity = name

        if min_dist > self.threshold:
            id = None
        else:
            id = identity

        return id, min_dist

    def img_path_to_encoding(self, image_path):
        img = cv2.imread(image_path, 1)
        return self.img_to_encoding(img)

    def img_to_encoding(self, image):
        image = cv2.resize(image, (96, 96))
        img = image[..., ::-1]
        img = np.around(np.transpose(img, (2, 0, 1)) / 255.0, decimals=12)
        x_train = np.array([img])
        embedding = self.model.predict_on_batch(x_train)
        return embedding

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
