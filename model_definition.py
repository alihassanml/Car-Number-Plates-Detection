
import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Model

def create_model():
    inception_rsnet = InceptionResNetV2(weights='imagenet',
                                         include_top=False,
                                         input_tensor=Input(shape=(224,224,3)))
    inception_rsnet.trainable = False
    
    headmodel = inception_rsnet.output
    headmodel = Flatten()(headmodel)
    headmodel = Dense(500, activation='relu')(headmodel)
    headmodel = Dense(250, activation='relu')(headmodel)
    headmodel = Dense(4, activation='sigmoid')(headmodel)
    
    model = Model(inputs=inception_rsnet.input, outputs=headmodel)
    return model
