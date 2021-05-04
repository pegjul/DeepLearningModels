#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras import layers as tfl
import tensorflow_addons as tfa
from tensorflow_addons import layers as tfal
import numpy as np
import pandas as pd
import random
from PIL import Image
import os


from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()


# Preprocess Data input

# In[2]:


df = pd.read_csv('~/kaggle_datasets/plant-pathology-2021-fgvc8/train.csv', index_col='image').to_dict()
data = df['labels']
keys=["healthy", "scab", "frog_eye_leaf_spot", "rust", "complex", "powdery_mildew"]

def text_to_vec(value, keys):
    out = np.zeros(len(keys), np.float32)
    for i, key in enumerate(keys):
        if key in value:
            out[i] = 1
    assert out.sum() > 0, print(value, out)
    return out

for img in data:
    data[img] = text_to_vec(data[img], keys)


# In[17]:


images = list(data.keys())
random.shuffle(images)
test_train_slpit = 5/95
img_shape = 256

def normalize(img):
    return img/255

def resize(img):
    long_side = max(img.size)
    ratio = img_shape / long_side
    new_shape = (int(img.width*ratio), int(img.height*ratio))
    return img.resize(new_shape)

def padding(img):
    height_pad = (img_shape - img.shape[0])/2
    height_pad_f = int(np.floor(height_pad))
    height_pad_c = int(np.ceil(height_pad))
    width_pad = (img_shape - img.shape[1])/2
    width_pad_f = int(np.floor(width_pad))
    width_pad_c = int(np.ceil(width_pad))
    pad = ((height_pad_f, height_pad_c), (width_pad_f, width_pad_c), (0, 0))
    return np.pad(img, pad, 'constant', constant_values=0.5)

def gen(start, end):
    for img in images[start:end]:
        I = Image.open(os.path.join("/home/julian/kaggle_datasets/plant-pathology-2021-fgvc8/train_images", img))
        I = resize(I)
        I = np.array(I, np.float32)
        I = normalize(I)
        I = padding(I)
        yield I, data[img]

        
batch_size = 32

output_signature=(tf.TensorSpec(shape=(img_shape, img_shape, 3), dtype=tf.float32), tf.TensorSpec(shape=(len(keys)), dtype=tf.float32))
train_start = 0
train_end = (int(len(images) * (1 - test_train_slpit))//batch_size)*batch_size
test_start = train_end
test_end = len(images) + 1

dataset_train = tf.data.Dataset.from_generator(gen, output_signature=output_signature, 
                  args=(train_start, train_end)).shuffle(16).repeat().cache().batch(batch_size)
dataset_test = tf.data.Dataset.from_generator(gen, output_signature=output_signature, 
                  args=(train_start, train_end)).repeat().cache().batch(batch_size)

steps_per_epoch_train = np.ceil(train_end/batch_size)
steps_per_epoch_test = np.ceil((test_end - test_start)/batch_size)


# Create Model

# In[4]:


num_heads = 4
transformer_layers = 8
mlp_head_units = [256, 128]  # Size of the dense layers of the final classifier

def residual(inp, filters):
    x = tfl.Conv2D(filters, 3, padding="same")(inp)
    x = tfal.GELU()(x)
    x = tfl.Conv2D(filters*2, 1)(x)
    x = tfal.GELU()(x)
    x = tfl.Concatenate()([x, inp])
    x = tfl.LayerNormalization(epsilon=1e-6)(x)
    x = tfl.Conv2D(filters, 3, 2, padding="same")(x)
    x = tfal.GELU()(x)
    return x

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = tfl.Dense(units, activation=tf.nn.gelu)(x)
        x = tfl.Dropout(dropout_rate)(x)
    return x

class PositionalEncoding(tfl.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PositionalEncoding, self).__init__()
        self.projection = tfl.Dense(units=projection_dim)
        self.position_embedding = tfl.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
        self.positions = tf.range(start=0, limit=num_patches, delta=1)

    def call(self, inputs):
        positional_encoding = self.projection(inputs) + self.position_embedding(self.positions)
        return positional_encoding

inp = tf.keras.Input(shape=(img_shape, img_shape, 3))
x = residual(inp, 8)
x = residual(x, 16)
x = residual(x, 32)
x = residual(x, 64)
x = tfl.Reshape((-1, 64))(x)
projection_dim = x.shape[-1]
num_patches = x.shape[-2]
x = PositionalEncoding(num_patches, projection_dim)(x)
transformer_units = [
    projection_dim * 2,
    projection_dim,
]

for _ in range(transformer_layers):
        x1 = tfl.LayerNormalization(epsilon=1e-6)(x)
        # Create a multi-head attention layer.
        attention_output = tfl.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = tfl.Add()([attention_output, x])
        # Layer normalization 2.
        x3 = tfl.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        x = tfl.Add()([x3, x2])

x = tfl.LayerNormalization(epsilon=1e-6)(x)
x = tfl.Flatten()(x)
x = tfl.Dropout(0.5)(x)
x = mlp(x, hidden_units=mlp_head_units, dropout_rate=0.5)
x = tfl.Dense(len(keys))(x)

model = tf.keras.Model(inp, x)


# In[5]:


learning_rate = 0.001
weight_decay = 0.0001
num_epochs = 100

optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)

loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# In[1]:


model.compile(optimizer=optimizer,
              loss=loss,
              metrics=[tf.keras.metrics.BinaryAccuracy()])

checkpoint_filepath = "tmp/checkpoint"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint( checkpoint_filepath, monitor="binary_accuracy", save_best_only=True, save_weights_only=True)
history = model.fit(dataset_train,
                    epochs=num_epochs,
                    steps_per_epoch=steps_per_epoch_train,
                    callbacks=[checkpoint_callback])


# In[ ]:


model.load_weights(checkpoint_filepath)

