from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import matplotlib.pyplot as plt
from args import image_shape
import tensorflow_addons as tfa


def image_processing(path):
    path = path
    x = image.load_img(path, target_size=(image_shape[0], image_shape[1]),color_mode='rgb')
    x = image.img_to_array(x)
    x = tf.expand_dims(x, axis=0)
    x /= 255.
    return x

# model = load_model('Generator.h5')
from tensorflow.keras.utils import CustomObjectScope
with CustomObjectScope({'Addons>InstanceNormalization':tfa.layers.InstanceNormalization}):
    model = load_model('Gy2x.h5')

x = image_processing('9999.jpg')
picture = model.predict(x)
picture = tf.squeeze(picture,axis=0)
picture = tf.keras.preprocessing.image.array_to_img(picture)
plt.imshow(picture)
# plt.savefig('img'+str(i))
plt.show()

