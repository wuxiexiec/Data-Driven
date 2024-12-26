from cyclegan import CycleGan
import args
from callback import Cyclegancbk
from generatorx2y import buildgeneratorx2y
from generatory2x import buildgeneratory2x
from discriminator import build_discriminator
import loss
from data import real_x,real_y
import tensorflow as tf

if __name__ == '__main__':
    generatorx2y = buildgeneratorx2y()
    generatory2x = buildgeneratory2x()
    discriminatorx = build_discriminator()
    discriminatory = build_discriminator()

    cyclegan = CycleGan(generatorx2y,generatory2x,discriminatorx,discriminatory)

    cyclegan.compile(generatorx2y_optimizer=loss.generatorx2y_optimizer,
                     generatory2x_optimizer=loss.generatory2x_optimizer,
                     discriminator_optimizer=loss.discriminator_optimizer,
                     identity_loss_fn=loss.identity_loss_fn,
                     recon_loss_fn=loss.recon_loss_fn,
                     adver_loss_fn=loss.adver_loss_fn,
                     dicriminator_loss_fn=loss.discriminator_loss_fn)

    cyclegancbk = Cyclegancbk()

    cyclegan.fit(tf.data.Dataset.zip((real_x,real_y)),batch_size=args.batch_size,epochs=args.epochs,callbacks=[cyclegancbk])
    # tf.data.Dataset.
    #save model
    generatorx2y.save('Gx2y.h5')
    generatory2x.save('Gy2x.h5')
    discriminatorx.save('Dx.h5')
    discriminatory.save('Dy.h5')
