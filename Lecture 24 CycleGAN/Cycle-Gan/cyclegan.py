import tensorflow as tf
import args

class CycleGan(tf.keras.Model):
    def __init__(self,generatorx2y,generatory2x,discriminatorx,discriminatory):
        super().__init__()
        self.generatorx2y = generatorx2y
        self.generatory2x = generatory2x
        self.discriminatorx = discriminatorx
        self.discriminatory = discriminatory


    def compile(self,generatorx2y_optimizer,
                generatory2x_optimizer,
                discriminator_optimizer,
                identity_loss_fn,
                recon_loss_fn,
                adver_loss_fn,
                dicriminator_loss_fn):
        super().compile()
        self.generatorx2y_optimizer = generatorx2y_optimizer
        self.generatory2x_optimizer = generatory2x_optimizer
        self.dx_optimizer = discriminator_optimizer
        self.dy_optimizer = discriminator_optimizer
        self.identity_loss_fn = identity_loss_fn
        self.recon_loss_fn = recon_loss_fn
        self.adver_loss_fn = adver_loss_fn
        self.discriminator_loss_fn = dicriminator_loss_fn

    def normalize_img(self,img):
        img = tf.cast(img, dtype=tf.float32)
        # Map values in the range [-1, 1]
        return (img / 127.5) - 1.0

    def train_step(self, batch_data):

        real_x,real_y = batch_data
        real_x = self.normalize_img(real_x)
        real_y = self.normalize_img(real_y)

        with tf.GradientTape(persistent=True) as tape:
            # get recon_x
            gen_y = self.generatorx2y(real_x,training = True)
            recon_x = self.generatory2x(gen_y,training = True)

            # get recon_y
            gen_x = self.generatory2x(real_y,training = True)
            recon_y = self.generatorx2y(gen_x,training = True)

            # get_discriminator_output
            dx_real = self.discriminatorx(real_x,training = True)
            dx_fake = self.discriminatorx(gen_x,training = True)

            dy_real = self.discriminatory(real_y,training = True)
            dy_fake = self.discriminatory(gen_y,training = True)

            dx_loss = self.discriminator_loss_fn(dx_real,dx_fake)
            dy_loss = self.discriminator_loss_fn(dy_real,dy_fake)


            real_label = tf.ones_like(dx_real)
            # fake_label = tf.zeros_like(dy_fake)

            #caculate x2y losses
            identity_loss_x2y = self.identity_loss_fn(gen_y,real_x)
            adver_loss_x2y = self.adver_loss_fn(real_label,dy_fake)
            recon_loss_x2y = self.recon_loss_fn(real_x,recon_x)
            total_loss_x2y = adver_loss_x2y + args.cycle_consistency_loss * recon_loss_x2y + args.identity_loss_weight * identity_loss_x2y

            #calculate y2x losses
            identity_loss_y2x = self.identity_loss_fn(gen_x,real_y)
            adver_loss_y2x = self.adver_loss_fn(real_label,dx_fake)
            recon_loss_y2x = self.recon_loss_fn(real_y,recon_y)
            total_loss_y2x = adver_loss_y2x + args.cycle_consistency_loss * recon_loss_y2x + args.identity_loss_weight * identity_loss_y2x


        gradient_x2y = tape.gradient(total_loss_x2y,self.generatorx2y.trainable_weights)
        self.generatorx2y_optimizer.apply_gradients(zip(gradient_x2y,self.generatorx2y.trainable_weights))
        gradient_y2x = tape.gradient(total_loss_y2x,self.generatory2x.trainable_weights)
        self.generatory2x_optimizer.apply_gradients(zip(gradient_y2x,self.generatory2x.trainable_weights))

        gradient_dx = tape.gradient(dx_loss,self.discriminatorx.trainable_weights)
        self.dx_optimizer.apply_gradients(zip(gradient_dx,self.discriminatorx.trainable_weights))
        gradient_dy = tape.gradient(dy_loss,self.discriminatory.trainable_weights)
        self.dy_optimizer.apply_gradients(zip(gradient_dy,self.discriminatory.trainable_weights))


        return {
            'Gx2y_total_loss':total_loss_x2y,
            'Gx2y_recon_loss':recon_loss_x2y,
            'Gx2y_adver_loss':adver_loss_x2y,
            'Dx_loss':dx_loss,

            'Gy2x_total_loss':total_loss_y2x,
            'Gy2x_recon_loss':recon_loss_y2x,
            'Gy2x_adver_loss':adver_loss_y2x,
            'Dy_loss':dy_loss
        }

    def call(self, inputs):
        # 由于我们不需要cyclegan类实例化模型做前向传播，所以这里随便写点什么就可以
        return inputs