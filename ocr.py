# coding=utf-8
from __future__ import print_function
from ctpn.boxprocess import sort_box, correct_box
import numpy as np
import cv2
from models.densenet import dense_cnn
from ctpn.ctpn.detectors import TextDetector
from ctpn.ctpn.other import draw_boxes
from ctpn.detector import load_tf_model, ctpn
import keras.backend.tensorflow_backend as K

# 标签库及session初始化
sess, saver, net = load_tf_model()
char = []
with open('./label/char_std_5990.txt') as f:
    for ch in f.readlines():
        ch = ch.decode('gb18030')
        ch = ch.strip('\r\n')
        char.append(ch)
char = char[1:] + [u'卍']
id_to_char = {i: j for i, j in enumerate(char)}


class Ocr(object):
    def __init__(self, text_process=False):
        self._text_detector = TextDetector()
        self._ocr_model = dense_cnn(len(char))
        self._ocr_model.load_weights('./weights/weights-densenet-04.hdf5')
        self.rec_results = None
        self._text_process = text_process

    def predict(self, image):
        self._image = np.asarray(image)
        self.__text_detection()
        self.__text_recognition()
        return (self._rec_results, self._boxes)

    def __text_detection(self):
        scores, self._boxes, self._image = ctpn(self._image, sess, net)
        self._boxes = self._text_detector.detect(
            self._boxes, scores[:, np.newaxis], self._image.shape[:2])
        self._boxes, self.img_drawed_boxes = draw_boxes(
            self._image,
            self._boxes,
            caption='im_name',
            wait=True,
            is_display=False)
        self._boxes, self._image = correct_box(
            sort_box(self._boxes), self._image, self._text_process)

    def __text_recognition(self):
        self._rec_results = []
        for box in self._boxes:
            test_img = self._image[box[1]:box[7], box[0]:box[6]]
            test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
            scale = test_img.shape[0] * 1.0 / 32
            w = test_img.shape[1] / scale
            w = int(w)
            test_img = cv2.resize(test_img, (w, 32))
            test_img = np.array(test_img).astype(np.float32) / 255.0 - 0.5
            X = test_img.reshape((32, w, 1))
            X = np.array([X])
            y_pred = self._ocr_model.predict(X)
            y_pred = y_pred[:, :, :]
            word = K.get_value(
                K.ctc_decode(
                    y_pred,
                    input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1],
                )[0][0])[:, :]
            words = u''.join([id_to_char[x] for x in word[0]])
            self._rec_results.append(words)
