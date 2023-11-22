python

class ImageSegmentation:
    def __init__(self, image_path, model_path):
        self.image_path = image_path
        self.model_path = model_path
        self.new_model = tf.keras.models.load_model(model_path)

    def check_version(self):
        if tf.__version__ != '2.3.0':
            print('警告，为了防止出现环境冲突，tensorflow(tensorflow-gpu)版本建议为2.3.0。')
        if matplotlib.__version__ != '3.5.1':
            print('警告，为了防止出现环境冲突，matplotlib版本建议为3.5.1。')
        if np.__version__ != '1.21.5':
            print('警告，为了防止出现环境冲突，numpy版本建议为1.21.5。')

    def load_dataset(self):
        images = glob.glob(self.image_path)
        anno = images
        dataset = tf.data.Dataset.from_tensor_slices((images, anno))
        train_count = len(images)
        data_train = dataset.take(train_count)
        return data_train

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

    def set_config(self, data_train):
        data_train = data_train.map(self.load_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        BATCH_SIZE = 32
        data_train = data_train.repeat().shuffle(100).batch(BATCH_SIZE)
        return data_train

    def predict(self):
        data_train = self.load_dataset()
        data_train = self.set_config(data_train)

        for image, mask in data_train.take(1):
            pred_mask = self.new_model.predict(image)
            pred_mask = tf.argmax(pred_mask, axis=-1)
            pred_mask = pred_mask[..., tf.newaxis]

            print(np.unique(pred_mask[0].numpy()))
            plt.figure(figsize=(10, 10))
            plt.subplot(1, 3, 1)
            plt.imshow(tf.keras.preprocessing.image.array_to_img(image[0]))
            plt.subplot(1, 3, 2)
            plt.imshow(tf.keras.preprocessing.image.array_to_img(mask[0]))
            plt.subplot(1, 3, 3)
            plt.imshow(tf.keras.preprocessing.image.array_to_img(pred_mask[0]))
            plt.show()
