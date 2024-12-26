import tensorflow as tf
import matplotlib.pyplot as plt
import args

domain_a_img_path = args.domain_x_test_img_path
domain_b_img_path = args.domain_y_test_img_path

class Cyclegancbk(tf.keras.callbacks.Callback):
    def __init__(self):
        self.img_x = tf.keras.preprocessing.image.load_img(domain_a_img_path,target_size=args.image_shape)
        self.img_y = tf.keras.preprocessing.image.load_img(domain_b_img_path,target_size=args.image_shape)

    def on_epoch_end(self, epoch, logs=None):
        # get x,convert x to y ,then reconstrcut y to x
        self.img_x = tf.keras.preprocessing.image.img_to_array(self.img_x)
        self.input_x = tf.expand_dims(self.img_x,axis=0)
        self.x2y_image = self.model.generatorx2y(self.input_x)
        self.reconstrcutedy2x = self.model.generatory2x(self.x2y_image)
        self.img_x = self.img_x / 255.
        self.img_x = tf.keras.preprocessing.image.array_to_img(self.img_x)
        #squeeze
        self.x2y_image = tf.squeeze(self.x2y_image,axis=0)
        self.x2y_image = self.x2y_image / 255.
        self.x2y_image = tf.keras.preprocessing.image.array_to_img(self.x2y_image)
        self.reconstrcutedy2x = tf.squeeze(self.reconstrcutedy2x,axis=0)
        self.reconstrcutedy2x = self.reconstrcutedy2x / 255.
        self.reconstrcutedy2x = tf.keras.preprocessing.image.array_to_img(self.reconstrcutedy2x)

        # get y ,convert y to x,then reconstruct x to y
        self.img_y = tf.keras.preprocessing.image.img_to_array(self.img_y)
        self.input_y = tf.expand_dims(self.img_y,axis=0)
        self.y2x_image = self.model.generatory2x(self.input_y)
        self.reconstrcutedx2y = self.model.generatorx2y(self.y2x_image)
        #squeeze
        self.y2x_image = tf.squeeze(self.y2x_image,axis=0)
        self.y2x_image = self.y2x_image / 255.
        self.y2x_image = tf.keras.preprocessing.image.array_to_img(self.y2x_image)
        self.reconstrcutedx2y = tf.squeeze(self.reconstrcutedx2y,axis=0)/255.
        self.reconstrcutedx2y = self.reconstrcutedx2y / 255.
        self.reconstrcutedx2y = tf.keras.preprocessing.image.array_to_img(self.reconstrcutedx2y)
        self.img_y = self.img_y / 255.
        self.img_y = tf.keras.preprocessing.image.array_to_img(self.img_y)

        #plt the results
        self.plot_img = [self.img_x,self.x2y_image,self.reconstrcutedy2x,
                         self.img_y,self.y2x_image,self.reconstrcutedx2y]

        self.plot_title = ['x','x2y','recony2x',
                           'y','y2x','reconx2y']

        for i in range(0,6):
            plt.subplot(2,3,i+1)
            plt.imshow(self.plot_img[i])
            plt.title(self.plot_title[i])
            plt.axis('off')

        plt.savefig("epc_{ep}.png".format(ep=epoch))
        plt.close()