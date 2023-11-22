import tensorflow as tf

# 压缩激活模块 (Squeeze-and-Excitation Module)
class SEBlock(tf.keras.layers.Layer):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc1 = tf.keras.layers.Dense(channels // reduction, activation='relu')
        self.fc2 = tf.keras.layers.Dense(channels, activation='sigmoid')

    def call(self, x):
        batch, height, width, channels = x.shape
        se = self.avg_pool(x)
        se = self.fc1(se)
        se = self.fc2(se)
        se = tf.reshape(se, (-1, 1, 1, channels))
        return x * se


# 空洞空间金字塔池化模块 (ASPP)
class ASPP(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(ASPP, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, 1, activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters, 3, dilation_rate=3, padding='same', activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(filters, 3, dilation_rate=5, padding='same', activation='relu')
        self.conv4 = tf.keras.layers.Conv2D(filters, 3, dilation_rate=7, padding='same', activation='relu')

    def call(self, x):
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        combine = tf.concat([feat1, feat2, feat3, feat4], axis=3)
        return combine


# 空间注意力模块 (Spatial Attention)
class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = tf.keras.layers.Conv2D(1, 7, padding='same', activation='sigmoid')

    def call(self, x):
        avg_feat = tf.reduce_mean(x, axis=3)
        max_feat = tf.reduce_max(x, axis=3)
        concat = tf.concat([avg_feat, max_feat], axis=3)
        sa = self.conv(concat)
        return x * sa


# 卷积块注意力模块 (CBAM)
class CBAM(tf.keras.layers.Layer):
    def __init__(self, channels):
        super(CBAM, self).__init__()
        self.se = SEBlock(channels)
        self.sa = SpatialAttention()

    def call(self, x):
        x = self.se(x)
        x = self.sa(x)
        return x


# 完整的 MA-Unet 模型
class MAUnet(tf.keras.Model):
    def __init__(self):
        super(MAUnet, self).__init__()
        # 编码器
        self.encoder_conv1 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')
        self.encoder_conv2 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')
        self.encoder_conv3 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')
        self.encoder_conv4 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')
        self.encoder_conv5 = tf.keras.layers.Conv2D(1024, 3, padding='same', activation='relu')

        self.cbam1 = CBAM(64)
        self.cbam2 = CBAM(128)
        self.cbam3 = CBAM(256)
        self.cbam4 = CBAM(512)

        self.aspp = ASPP(1024)

        # 解码器
        self.decoder_conv1 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')
        self.decoder_conv2 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')
        self.decoder_conv3 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')
        self.decoder_conv4 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')

        self.output_conv = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')

    def call(self, x):
        # 编码器
        x1 = self.encoder_conv1(x)
        x1 = self.cbam1(x1)

        x2 = self.encoder_conv2(x1)
        x2 = self.cbam2(x2)

        x3 = self.encoder_conv3(x2)
        x3 = self.cbam3(x3)

        x4 = self.encoder_conv4(x3)
        x4 = self.cbam4(x4)

        x5 = self.encoder_conv5(x4)
        x5 = self.aspp(x5)

        # 解码器
        x6 = self.decoder_conv1(x5)
        x7 = self.decoder_conv2(x6)
        x8 = self.decoder_conv3(x7)
        x9 = self.decoder_conv4(x8)

        out = self.output_conv(x9)

        return out
