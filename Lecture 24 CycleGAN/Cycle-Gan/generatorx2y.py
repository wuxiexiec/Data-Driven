from tensorflow.keras import Model,layers
import args
from tensorflow_addons.layers import  InstanceNormalization

def buildgeneratorx2y():
    model_input = layers.Input(shape=(args.image_shape))
    # layer1
    layer1 = layers.Conv2D(128,2,1,padding='same')(model_input)
    layer1 = InstanceNormalization()(layer1)
    layer1 = layers.LeakyReLU()(layer1)
    # layer2
    layer2 = layers.Conv2D(128,2,1,padding='same')(layer1)
    layer2 = InstanceNormalization()(layer2)
    layer2 = layers.LeakyReLU()(layer2)
    # layer3
    layer3 = layers.Conv2D(256,2,1,padding='same')(layer2)
    layer3 = InstanceNormalization()(layer3)
    layer3 = layers.LeakyReLU()(layer3)
    # decoder
    layer4 = layers.Conv2DTranspose(256,2,1,padding='same')(layer3)
    layer4 = InstanceNormalization()(layer4)
    layer4 = layers.LeakyReLU()(layer4)
    layers.add([layer4,layer3])
    # layer5
    layer5 = layers.Conv2DTranspose(128,2,1,padding='same')(layer4)
    layer5 = InstanceNormalization()(layer5)
    layer5 = layers.LeakyReLU()(layer5)
    layers.add([layer5,layer2])
    # layer6
    layer6 = layers.Conv2DTranspose(128, 2, 1, padding='same')(layer5)
    layer6 = InstanceNormalization()(layer6)
    layer6 = layers.LeakyReLU()(layer6)

    model_output = layers.Conv2DTranspose(3,1,1)(layer6)

    generatorx2y = Model(model_input,model_output,name='Gx2y')

    return generatorx2y

if __name__ == '__main__':
    encoderx2y = buildgeneratorx2y()
    encoderx2y.summary()