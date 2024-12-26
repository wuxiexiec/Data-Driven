import tensorflow as tf
from tensorflow.keras import layers
import args


def build_discriminator():
    input = layers.Input(shape=args.image_shape)
    x = layers.Conv2D(128,2,1,padding='same')(input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256,2,1,padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256,2,1,padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    output = layers.Conv2D(1,3,3,activation='sigmoid')(x)

    discriminator = tf.keras.Model(input,output,name='discriminator')

    return discriminator


if __name__ == '__main__':
    discriminator = build_discriminator()
    discriminator.summary()