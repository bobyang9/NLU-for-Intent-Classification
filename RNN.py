import collections
import pathlib
import re
import string
import os
import time
import numpy as np
import sklearn
import scipy
import matplotlib.pyplot as plt

from load_data import *

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras import utils
from tensorflow.keras import regularizers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Embedding
from tensorflow import keras

def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])

# Loading data
TRAIN_PATH = './snips/snips_train_actual.csv'
TEST_PATH = './snips/snips_test_actual.csv'

data_loader = DataLoader(TRAIN_PATH, None, TEST_PATH)
data_loader.split_train_valid(valid_size=0.05, keep_class_ratios=True)

X_train, y_train = data_loader.get_train_data()
X_valid, y_valid = data_loader.get_valid_data()

num_classes = len(set(y_train))

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

feature_extractor = FeatureExtractor2(X_train, X_valid)
feature_extractor.extract_features()
X_train = feature_extractor.get_train_encodings()
X_valid = feature_extractor.get_valid_encodings()

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

num_features = feature_extractor.vocab_size+1 # plus 1 is for UNK
print(num_features)

# Loading embedding
embedding_matrix = np.eye(num_features)

# Model
embedding_layer = Embedding(num_features, num_features, embeddings_initializer=keras.initializers.Constant(embedding_matrix), trainable=False)

int_sequences_input = keras.Input(shape=(feature_extractor.max_dim,), dtype="int64")
embedded_sequences = embedding_layer(int_sequences_input)
x = layers.Bidirectional(tf.keras.layers.LSTM(64))(embedded_sequences)
x = layers.Dropout(0.2)(x)
x = layers.Dense(64, activation='relu')(x)
preds = layers.Dense(num_classes, activation="softmax")(x)
model = keras.Model(int_sequences_input, preds)
model.summary()

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(1e-4),metrics=[keras.metrics.SparseCategoricalAccuracy()])

checkpoint_path = "rnn_take_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

# Training
history = model.fit(train_dataset, epochs=20, validation_data=val_dataset, callbacks=[cp_callback])

# Evaluation
val_loss, val_acc = model.evaluate(val_dataset)

print("val loss is %f"%(val_loss))
print("val accuracy is %f"%(val_acc))

print(history.history.keys())

# Plotting
plt.figure(figsize=(16,8))

plt.subplot(1,2,1)
plot_graphs(history, 'sparse_categorical_accuracy')
plt.ylim(None,1)

plt.subplot(1,2,2)
plot_graphs(history, 'loss')
plt.ylim(0,None)

plt.savefig("accuracy_loss_plot.png")
plt.show()