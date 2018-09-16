import matplotlib as mpl

# This line allows mpl to run with no DISPLAY defined
#mpl.use('Agg')

import numpy
#import preprocess
#import matplotlib.pyplot as plt
#import matplotlib as mpl
import gzip
import pickle
from keras.layers import Flatten, Dropout, LeakyReLU, Input, Activation
from keras.models import Model
from keras.layers.convolutional import UpSampling2D
from keras.optimizers import Adam
from keras.datasets import mnist
import pandas as pd
import numpy as np
import keras.backend as K
from keras_adversarial.legacy import Dense, BatchNormalization, Convolution2D
from keras_adversarial.image_grid_callback import ImageGridCallback
from keras_adversarial import AdversarialModel, simple_gan, gan_targets
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling
from image_utils import dim_ordering_fix, dim_ordering_input, dim_ordering_reshape, dim_ordering_unfix
from scipy.ndimage.filters import gaussian_filter


def leaky_relu(x):
    return K.relu(x, 0.2)


def model_generator():
    nch = 64
    g_input = Input(shape=[400])
    H = Dense(nch * 46 * 46)(g_input)
    H = BatchNormalization(mode=2)(H)
    H = Activation('relu')(H)
    H = dim_ordering_reshape(nch, 46)(H)
    H = UpSampling2D(size=(2, 2))(H)
    H = Convolution2D(int(nch / 2), 3, 3, border_mode='same')(H)
    H = BatchNormalization(mode=2, axis=1)(H)
    H = Activation('relu')(H)
    H = Convolution2D(int(nch / 4), 3, 3, border_mode='same')(H)
    H = BatchNormalization(mode=2, axis=1)(H)
    H = Activation('relu')(H)
    H = Convolution2D(1, 1, 1, border_mode='same')(H)
    g_V = Activation('sigmoid')(H)
    return Model(g_input, g_V)


def model_discriminator(input_shape=(1, 92, 92), dropout_rate=0.5):
    d_input = dim_ordering_input(input_shape, name="input_x")
    nch = 64
    # nch = 128
    H = Convolution2D(int(nch / 2), 5, 5, subsample=(2, 2), border_mode='same', activation='relu')(d_input)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    H = Convolution2D(nch, 5, 5, subsample=(2, 2), border_mode='same', activation='relu')(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    H = Flatten()(H)
    H = Dense(int(nch / 2))(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    d_V = Dense(1, activation='sigmoid')(H)
    return Model(d_input, d_V)


def mnist_process(x):
    x = x.astype(np.float32) / 255.0
    return x


def mnist_data():
    (xtrain, ytrain), (xtest, ytest) = mnist.load_data()
    return mnist_process(xtrain), mnist_process(xtest)


def generator_sampler(latent_dim, generator):
    def fun():
        zsamples = np.random.normal(size=(10 * 10, latent_dim))
        gen = dim_ordering_unfix(generator.predict(zsamples))
        return gen.reshape((10, 10, 92, 92))

    return fun

def sum(x, i, j):
    sum = (x[i, j] + x[i-1, j-1] + x[i-1, j] + x[i-1, j+1] + x[i, j - 1] + x[i, j + 1] + x[i+1, j-1] + x[i+1, j] + x[i+1, j+1])
    return sum
    print sum
def lissage(x):#seulement pour les carre!!
    y = np.zeros((np.size(x[0]), np.size(x[0])))
    for i in range(0, np.size(x[0])):
        y[0][i] = 9*x[0, i]
        y[i, 0] = 9*x[i, 0]
        y[i, (np.size(x[0]) - 1)] = 9*x[i, (np.size(x[0]) - 1)]
        y[(np.size(x[0]) - 1), i] = 9*x[(np.size(x[0]) - 1), i]

    for i in range((np.size(x[0]) - 2), 0, -1):
        for j in range((np.size(x[0]) - 2), 0, -1):
            y[i, j] = sum(x, i, j) 

            
    mini = np.min(y.ravel())
    maxi = np.max(y.ravel())
    
    y = (y - mini)/(maxi-mini)
    
    return y

# Lissage d'une matrice ypred, ytrue par filtre gaussien
ypred_blur = gaussian_filter(ypred, sigma=3, order=order, output=None, mode='reflect', cval=0.0, truncate=4.0)
ytrue_blur = gaussian_filter(ytrue, sigma=3, order=order, output=None, mode='reflect', cval=0.0, truncate=4.0)
dypred = np.subtract(ypred, 10*ypred_blur)

def generator_skampler(latent_dim, generator):
    zsamples = np.random.normal(size=(10 * 10, latent_dim))
    gen = dim_ordering_unfix(generator.predict(zsamples))
    return gen.reshape((10, 10, 92, 92))

    
if __name__ == "__main__":
    # z \in R^100
    input_shape = (1, 92, 92)
    latent_dim = 400
    generator = model_generator()
    discriminator = model_discriminator(input_shape=input_shape)
    
    generator.load_weights('./GAN/generator.h5')
    discriminator.load_weights('./GAN/discriminator.h5')

    fname = "base_hiver_2008.pklgz"
    with gzip.open(fname, "rb") as fp:
        dictio = pickle.load(fp)
    
    data = dictio['SSTMW']
    x = data[:, :92, :92].astype(np.float32) 
    xtrain = x[:-10]
    
    mini = np.min(xtrain.ravel())
    maxi = np.max(xtrain.ravel())
    xtrain = (xtrain - mini)/(maxi-mini)
       
    yy = generator_skampler(latent_dim, generator)
    
    y = lissage(yy[5][5])
    print y 
    mpl.pyplot.imshow(y)
    mpl.pyplot.savefig('Image_lisser_sans_convolution')
    mpl.pyplot.show()
    
    yy = yy * (maxi-mini) + mini
    #print discriminator(yy[0][0])


    mpl.pyplot.imshow(yy[2][0])
    mpl.pyplot.savefig('Image_generer_sans_lissage_sans_convolution')
    mpl.pyplot.colorbar()
    mpl.pyplot.show()

    mpl.pyplot.imshow(yy[3][5])
    mpl.pyplot.colorbar()
    mpl.pyplot.show()
   
