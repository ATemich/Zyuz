# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from .. import FaceRecognizer
from .facenet import prewhiten, to_rgb
import tensorflow as tf
import numpy as np
import cv2
import pickle
import os

class Identifier:
    def __init__(self, classifier_model):
        with open(classifier_model, 'rb') as infile:
            self.model, self.class_names = pickle.load(infile)

    def identify(self, embedding):
        if embedding is not None:
            predictions = self.model.predict_proba([embedding])
            print(predictions)
            best_class_indices = np.argmax(predictions, axis=1)
            return self.class_names[best_class_indices[0]]

class ClsFaceNet(FaceRecognizer):
    def __init__(self):
        super().__init__()

        path = os.path.dirname(os.path.realpath(__file__))
        model_name = '20180402-114759'
        model_dir = os.path.join(path, 'ckpt', model_name)
        graph_fr = tf.Graph()
        self.sess = tf.Session(graph=graph_fr)

        with graph_fr.as_default():
            saverf = tf.train.import_meta_graph(os.path.join(model_dir, f'model-{model_name}.meta'))
            saverf.restore(self.sess, os.path.join(model_dir, f'model-{model_name}.ckpt-275'))
        self.identifier = Identifier(os.path.join(path, 'test_classifier.pkl'))

    def load_from_pickled(self, path):
        with open(path, 'rb') as f:
            self.database.update(pickle.load(f))

    def recognize_image(self, image):
        encoding = self.img_to_encoding(image)
        id = self.identifier.identify(encoding)
        return id, 0

    def img_path_to_encoding(self, image_path):
        img = cv2.imread(image_path, 1)
        return self.img_to_encoding(img)

    def img_to_encoding(self, image):
        images_placeholder = self.sess.graph.get_tensor_by_name("input:0")
        images_placeholder = tf.image.resize_images(images_placeholder, (160, 160))
        embeddings = self.sess.graph.get_tensor_by_name("embeddings:0")
        phase_train_placeholder = self.sess.graph.get_tensor_by_name("phase_train:0")

        image = cv2.resize(image, (160, 160))
        image = prewhiten(image)

        feed_dict = {images_placeholder: [image], phase_train_placeholder: False}
        return self.sess.run(embeddings, feed_dict=feed_dict)[0]
