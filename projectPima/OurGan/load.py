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
    nch = 128
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
    nch = 128
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

def lissage_median(M):
    shape = M.shape
    
    n_pixel = np.zeros((9))
    
    for i in range(shape[0]-1):
        for j in range(shape[1]-1):
            if j > 0 and i > 0:
                n_pixel[0] = M[i-1,j-1]
                n_pixel[1] = M[i-1,j]
                n_pixel[2] = M[i-1,j+1]
                n_pixel[3] = M[i,j-1]
                n_pixel[4] = M[i,j]
                n_pixel[5] = M[i,j+1]
                n_pixel[6] = M[i+1,j-1]
                n_pixel[7] = M[i+1,j]
                n_pixel[8] = M[i+1,j+1]
                s = np.sort(n_pixel, axis=None)  
                M[i,j] = s[4]
    return M

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
    
    generator.load_weights('./output/gan_convolutional/generator.h5')
    discriminator.load_weights('./output/gan_convolutional/discriminator.h5')
    
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
    
    yy = yy * (maxi-mini) + mini
    
    ylm = yy[1][3]
    mpl.pyplot.imshow(ylm)
    mpl.pyplot.savefig('Image_128_generer_sans_lissage')
    mpl.pyplot.colorbar()
    mpl.pyplot.show()

    
    y_median = lissage_median(yy[1][3])
    mpl.pyplot.imshow(y_median)
    mpl.pyplot.savefig('Image_128_generer_avec_lissage_Median')
    mpl.pyplot.colorbar()
    mpl.pyplot.show()

    y_filtred = gaussian_filter(lissage_median(yy[1][4]), sigma=2, output=None, mode='reflect', cval=0.0, truncate=4.0)
    mpl.pyplot.imshow(y_filtred)
    mpl.pyplot.savefig('Image_128_generer_avec_lissage_GaussienPlusMedian')
    mpl.pyplot.colorbar()
    mpl.pyplot.show()
        
    y_gauss1 = gaussian_filter(yy[1][3], sigma=2, output=None, mode='reflect', cval=0.0, truncate=4.0)
    mpl.pyplot.imshow(y_gauss1)
    mpl.pyplot.savefig('Image_128_generer_avec_lissage_gaussien_ss')
    mpl.pyplot.colorbar()
    mpl.pyplot.show()

    
    y_filtred2 = lissage_median(gaussian_filter(yy[8][0], sigma=2, output=None, mode='reflect', cval=0.0, truncate=4.0))
    mpl.pyplot.imshow(y_filtred2)
    mpl.pyplot.savefig('Image_128_generer_avec_lissage_MedianPlusGaussin')
    mpl.pyplot.colorbar()
    mpl.pyplot.show()
    
    y_gauss2 = gaussian_filter(yy[1][0], sigma=3, output=None, mode='reflect', cval=0.0, truncate=4.0)
    mpl.pyplot.imshow(y_gauss2)
    mpl.pyplot.savefig('Image_128_generer_avec_lissage_gaussien_Sigma3')
    mpl.pyplot.colorbar()
    mpl.pyplot.show()


    
