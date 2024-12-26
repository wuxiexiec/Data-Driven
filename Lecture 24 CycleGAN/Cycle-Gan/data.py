from tensorflow.keras.preprocessing import image_dataset_from_directory
import args
import tensorflow as tf
import matplotlib.pyplot as plt

x_path = args.x_path
y_path = args.y_path

def normalize_img(img):
    img = tf.cast(img, dtype=tf.float32)
    # Map values in the range [-1, 1]
    return (img / 127.5) - 1.0



real_x = image_dataset_from_directory(directory=x_path,
                                          label_mode=None,
                                          batch_size=args.batch_size,
                                          color_mode= 'rgb',
                                          image_size=(args.image_shape[0],args.image_shape[1]),
                                          shuffle=True)

real_y = image_dataset_from_directory(directory=y_path,
                                      label_mode=None,
                                      batch_size=args.batch_size,
                                      color_mode='rgb',
                                      image_size=(args.image_shape[0], args.image_shape[1]),
                                      shuffle=True
                                      )



if __name__ == '__main__':
    print(real_x_data)
