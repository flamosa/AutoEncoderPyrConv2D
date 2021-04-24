import tensorflow as t
import tensorflow.keras as k
from tensorflow.keras import layers
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.backend import dropout


#class Cluster2D( object ):
#    __init__(self, inX, inY ):
#        input_shape = (inX,inY, 1)
#        input_layer = Input(shape=input_shape, batch_size=None)
#        layer_c1 = layers.Conv2D

class Encoder(k.layers.Layer):
    def __init__(self, inX, inY, kX_0, kY_0, nfilters, drop_rate=0.5, drop_seed=20 ):
            super(Encoder,self).__init__()
            self.convolve1=layers.Conv2D (  filters=nfilters,
                                            kernel_size=(kX_0,kY_0),
                                            strides=(1,1),
                                            padding='valid',
                                            data_format="channels_last",
                                            activation='relu',
                                            use_bias=True)

            self.encode_drop = layers.Dropout(rate = drop_rate, seed = drop_seed )
            self.encode_subsample_1 = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid", data_format="channels_last")

            self.convolve2=layers.Conv2D (  filters=nfilters,
                                            kernel_size=(kX_0-2, kY_0-2),
                                            strides=(1,1),
                                            padding='same',
                                            data_format="channels_last",
                                            activation='relu',
                                            use_bias = True)

    def call(self, input_features):
        x = self.convolve1(input_features)
        x = self.encode_drop(x)
        x = self.convolve2(input_features)
        return x


class Decoder(k.layers.Layer):
    def __init__(self, nfilters, outX, outY, drop_rate, drop_seed, kX_out, kY_out ):
        super(Decoder, self).__init__()
        self.decode_convolve1 = layers.Conv2D( filters=nfilters,
                                               kernel_size=(kX_out, kY_out),
                                               strides=(1,1),
                                               padding='same',
                                               data_format="channels_last",
                                               activation='relu',
                                               use_bias=True)

        self.drop_decode = layers.Dropout(rate=drop_rate, seed=drop_seed)
        self.decode_upsample_1 = layers.UpSampling2D(size=(2,2),data_format="channels_last", interpolation='bilinear')

        self.convolve2 = layers.Conv2D( filters=nfilters,
                                        kernel_size=(kX_out+2, kY_out+2),
                                        strides=(1,1),
                                        padding='same',
                                        data_format="channels_last",
                                        activation = 'relu',
                                        use_bias=True)

    def call(self, encoded):
        x = self.convolve1(encoded)
        x = self.drop_decode(x)
        x = self.convolve2(x)
        return x


class Autoencoder(k.Model):
    def __init__(self, nFilters, imSX, imSY, kX, kY, dropRate=0.5, dropSeed=20 ):

        t.config.experimental.set_memory_growth = True

        super(Autoencoder, self).__init__()
        self.loss = []
        self.encoder = Encoder(nfilters=nFilters, inX=imSX, inY = imSY, kX_0=kX, kY_0=kY, drop_rate=dropRate, drop_seed=dropSeed)
        self.decoder = Decoder(nfilters = nFilters, outX = imSX, outY = imSY, kX_out = kX, kY_out = kY, drop_rate=dropRate, drop_seed = dropSeed )

    def call(self, input_features):
        encoded = self.encoder(input_features)
        reconstructed = self.decoder(encoded)
        return reconstructed

    def get_model( self ):
        return Autoencoder

