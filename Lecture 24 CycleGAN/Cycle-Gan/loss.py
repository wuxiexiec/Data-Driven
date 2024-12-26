import tensorflow.keras as keras
import tensorflow as tf
import args

identity_loss_fn = keras.losses.MeanAbsoluteError()
recon_loss_fn = keras.losses.MeanAbsoluteError()
adver_loss_fn = keras.losses.MeanSquaredError()

def discriminator_loss_fn(real, fake):
    real_loss = keras.losses.mae(tf.ones_like(real), real)
    fake_loss = keras.losses.mae(tf.zeros_like(fake), fake)

    return (real_loss + fake_loss) * 0.5

generatorx2y_optimizer = tf.keras.optimizers.Adam(args.gx2y_learning_rate,0.9)
generatory2x_optimizer = tf.keras.optimizers.Adam(args.gy2x_learning_rate,0.9)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=args.d_learning_rate,beta_1=0.9)