#File per definire una rete neurale dati i numero di input e di output.
import tensorflow as tf
from keras import layers

#Creiamo una classe che estende tf.keras.Model per definire la nostra rete neurale.
class ReteNeurale(tf.keras.Model):
    def __init__(self, dim_in,dim_out,activation_output=None,name="rete", **kwargs):
        super(ReteNeurale, self).__init__(name=name,**kwargs)
        self.name=name
        self.cnn=tf.keras.Sequential(
            [
                layers.Conv2D(filters=32, kernel_size=(3,3),strides=1,padding="valid", activation='relu', kernel_initializer="glorot_uniform", input_shape=dim_in, name=self.name+"_conv1"),
                layers.MaxPooling2D((3, 2),data_format="channels_last",name=self.name+"_pool1"),
                layers.Conv2D(filters=32, kernel_size=(3,3),strides=1,padding="valid", activation='relu', kernel_initializer="glorot_uniform",name=self.name+"_conv2"),
                layers.MaxPooling2D((3, 2),data_format="channels_last",name=self.name+"_pool2"),
                layers.Conv2D(filters=32, kernel_size=(3,3),strides=1,padding="valid", activation='relu', kernel_initializer="glorot_uniform",name=self.name+"_conv3"),
                layers.MaxPooling2D((3, 2),data_format="channels_last",name=self.name+"_pool3"),
                layers.Flatten(data_format="channels_last",name=self.name+"_flatten"),
                layers.Dense(128, activation='relu', kernel_initializer="glorot_uniform",name=self.name+"_dense"),
                layers.Dense(32, activation='relu', kernel_initializer="glorot_uniform", bias_initializer="zeros",name=self.name+"_dense3")
                
            ]
        )
        self.lastLayer=layers.Dense(dim_out, activation=activation_output, kernel_initializer="glorot_uniform", bias_initializer="zeros",name=self.name+"_dense4")
    def call(self, inputs):
        #if inputs has 64,64,3 shape then it's an image and we need to reshape it to 1,64,64,3
        if len(inputs.shape)==3:
            inputs=tf.reshape(inputs,[1,inputs.shape[0],inputs.shape[1],inputs.shape[2]])


        #Controllo che l'input sia un tensore.
        if not isinstance(inputs, tf.Tensor):
            inputs = tf.convert_to_tensor(inputs)

        #input normalization
        inputs = tf.cast(inputs, tf.float32) / 255.0
        
        #x = self.conv1(inputs)
        #x = self.pool1(x)
        #x = self.conv2(x)
        #x = self.pool2(x)
        #x = self.conv3(x)
        #x = self.pool3(x)
        #x = self.flatten(x)
        #x = self.dense(x)
        #x = self.dense3(x)
        #x=self.dense4(x)
        #return x
        x=self.cnn.call(inputs)
        x=self.lastLayer(x)
        return x

class Actor(tf.keras.Model):
    def __init__(self,name, **kwargs):
        super(Actor, self).__init__(**kwargs)
        self.name=name
        self.cnn=tf.keras.Sequential(
            [
                layers.Conv2D(filters=32, kernel_size=(3,3),strides=1,padding="valid", activation='relu', kernel_initializer="glorot_uniform", input_shape=(64,64,3), name=self.name+"_conv1"),
                layers.MaxPooling2D((3, 2),data_format="channels_last",name=self.name+"_pool1"),
                layers.Conv2D(filters=32, kernel_size=(3,3),strides=1,padding="valid", activation='relu', kernel_initializer="glorot_uniform",name=self.name+"_conv2"),
                layers.MaxPooling2D((3, 2),data_format="channels_last",name=self.name+"_pool2"),
                layers.Conv2D(filters=32, kernel_size=(3,3),strides=1,padding="valid", activation='relu', kernel_initializer="glorot_uniform",name=self.name+"_conv3"),
                layers.MaxPooling2D((3, 2),data_format="channels_last",name=self.name+"_pool3"),
                layers.Flatten(data_format="channels_last",name=self.name+"_flatten"),
                layers.Dense(128, activation='relu', kernel_initializer="glorot_uniform",name=self.name+"_dense"),
                #layers.Dense(32, activation='relu', kernel_initializer="glorot_uniform", bias_initializer="zeros",name=self.name+"_dense3")
                
            ]
        )
        self.lastLayer=layers.Dense(15, kernel_initializer="glorot_uniform", bias_initializer="zeros",name=self.name+"_dense4")
    def call(self, inputs):
        #if inputs has 64,64,3 shape then it's an image and we need to reshape it to 1,64,64,3
        if len(inputs.shape)==3:
            inputs=tf.reshape(inputs,[1,inputs.shape[0],inputs.shape[1],inputs.shape[2]])


        #Controllo che l'input sia un tensore.
        if not isinstance(inputs, tf.Tensor):
            inputs = tf.convert_to_tensor(inputs)

        #input normalization
        inputs = tf.cast(inputs, tf.float32) / 255.0
        
        x=self.cnn.call(inputs)
        x=self.lastLayer(x)
        return x

class Critic(tf.keras.Model):
    def __init__(self,name,**kwargs):
        super(Critic, self).__init__(**kwargs)
        self.name=name
        self.cnn=tf.keras.Sequential(
            [
                layers.Conv2D(filters=32, kernel_size=(3,3),strides=1,padding="valid", activation='relu', kernel_initializer="glorot_uniform", input_shape=(64,64,3), name=self.name+"_conv1"),
                layers.MaxPooling2D((3, 2),data_format="channels_last",name=self.name+"_pool1"),
                layers.Conv2D(filters=32, kernel_size=(3,3),strides=1,padding="valid", activation='relu', kernel_initializer="glorot_uniform",name=self.name+"_conv2"),
                layers.MaxPooling2D((3, 2),data_format="channels_last",name=self.name+"_pool2"),
                layers.Conv2D(filters=32, kernel_size=(3,3),strides=1,padding="valid", activation='relu', kernel_initializer="glorot_uniform",name=self.name+"_conv3"),
                layers.MaxPooling2D((3, 2),data_format="channels_last",name=self.name+"_pool3"),
                layers.Flatten(data_format="channels_last",name=self.name+"_flatten"),
                layers.Dense(128, activation='relu', kernel_initializer="glorot_uniform",name=self.name+"_dense"),
                #layers.Dense(32, activation='relu', kernel_initializer="glorot_uniform", bias_initializer="zeros",name=self.name+"_dense3")
            ]
        )
        self.lastLayer=layers.Dense(1, kernel_initializer="glorot_uniform", bias_initializer="zeros",name=self.name+"_dense4")
    def call(self, inputs):
        #if inputs has 64,64,3 shape then it's an image and we need to reshape it to 1,64,64,3
        if len(inputs.shape)==3:
            inputs=tf.reshape(inputs,[1,inputs.shape[0],inputs.shape[1],inputs.shape[2]])

        #Controllo che l'input sia un tensore.
        if not isinstance(inputs, tf.Tensor):
            inputs = tf.convert_to_tensor(inputs)

        #input normalization
        inputs = tf.cast(inputs, tf.float32) / 255.0
        x=self.cnn.call(inputs)
        x=self.lastLayer(x)
        return x