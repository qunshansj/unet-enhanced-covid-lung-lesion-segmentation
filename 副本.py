python
import tensorflow as tf
import cv2
import numpy as np
import glob

class Segmentation:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
    
    def read_jpg(self, path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        return img

    def read_png(self, path):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=1)
        return img

    def normal_img(self, input_image, input_anno):
        input_image = tf.cast(input_image, tf.float32)
        input_image = input_image / 127.5 - 1
        input_anno -= 1
        return input_image, input_anno

    def load_images(self, input_images_path, input_anno_path):
        input_image = self.read_jpg(input_images_path)
        input_anno = self.read_png(input_anno_path)
        input_image = tf.image.resize(input_image, (224, 224))
        input_anno = tf.image.resize(input_anno, (224, 224))
        return self.normal_img(input_image, input_anno)

    def segment(self, image_path):
        image = self.load_images(image_path, image_path)
        pred_mask = self.model.predict(image)
        pred_mask = tf.argmax(pred_mask, axis=-1)
        pred_mask = pred_mask[..., tf.newaxis]
        return pred_mask.numpy()[0]
