import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import model_from_yaml
from tensorflow.keras.models import model_from_config
from tensorflow.keras.models import clone_model
from tensorflow.keras.models import save_model
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import constraints
from tensorflow.keras import activations


# Build GRU model

def build_model(input_shape, output_shape, num_layers, num_units, dropout, learning_rate, l2_reg, activation, batch_norm, batch_size, num_classes):
    model = Sequential()
    model.add(GRU(num_units, input_shape=input_shape, return_sequences=True, activation=activation, kernel_regularizer=regularizers.l2(l2_reg)))
    if batch_norm:
        model.add(BatchNormalization())
    if dropout > 0:
        model.add(Dropout(dropout))
    for i in range(num_layers-1):
        model.add(GRU(num_units, return_sequences=True, activation=activation, kernel_regularizer=regularizers.l2(l2_reg)))
        if batch_norm:
            model.add(BatchNormalization())
        if dropout > 0:
            model.add(Dropout(dropout))
    model.add(GRU(num_units, return_sequences=False, activation=activation, kernel_regularizer=regularizers.l2(l2_reg)))
    if batch_norm:
        model.add(BatchNormalization())
    if dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation='softmax'))
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Build LSTM model

def build_model(input_shape, output_shape, num_layers, num_units, dropout, learning_rate, l2_reg, activation, batch_norm, batch_size, num_classes):
    model = Sequential()
    model.add(LSTM(num_units, input_shape=input_shape, return_sequences=True, activation=activation, kernel_regularizer=regularizers.l2(l2_reg)))
    if batch_norm:
        model.add(BatchNormalization())
    if dropout > 0:
        model.add(Dropout(dropout))
    for i in range(num_layers-1):
        model.add(LSTM(num_units, return_sequences=True, activation=activation, kernel_regularizer=regularizers.l2(l2_reg)))
        if batch_norm:
            model.add(BatchNormalization())
        if dropout > 0:
            model.add(Dropout(dropout))
    model.add(LSTM(num_units, return_sequences=False, activation=activation, kernel_regularizer=regularizers.l2(l2_reg)))
    if batch_norm:
        model.add(BatchNormalization())
    if dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation='softmax'))
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Build RNN model

def build_model(input_shape, output_shape, num_layers, num_units, dropout, learning_rate, l2_reg, activation, batch_norm, batch_size, num_classes):
    model = Sequential()
    model.add(SimpleRNN(num_units, input_shape=input_shape, return_sequences=True, activation=activation, kernel_regularizer=regularizers.l2(l2_reg)))
    if batch_norm:
        model.add(BatchNormalization())
    if dropout > 0:
        model.add(Dropout(dropout))
    for i in range(num_layers-1):
        model.add(SimpleRNN(num_units, return_sequences=True, activation=activation, kernel_regularizer=regularizers.l2(l2_reg)))
        if batch_norm:
            model.add(BatchNormalization())
        if dropout > 0:
            model.add(Dropout(dropout))
    model.add(SimpleRNN(num_units, return_sequences=False, activation=activation, kernel_regularizer=regularizers.l2(l2_reg)))
    if batch_norm:
        model.add(BatchNormalization())
    if dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation='softmax'))
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Build multimodal model

def build_model(input_shape, output_shape, num_layers, num_units, dropout, learning_rate, l2_reg, activation, batch_norm, batch_size, num_classes):
    model = Sequential()
    model.add(GRU(num_units, input_shape=input_shape, return_sequences=True, activation=activation, kernel_regularizer=regularizers.l2(l2_reg)))
    if batch_norm:
        model.add(BatchNormalization())
    if dropout > 0:
        model.add(Dropout(dropout))
    for i in range(num_layers-1):
        model.add(GRU(num_units, return_sequences=True, activation=activation, kernel_regularizer=regularizers.l2(l2_reg)))
        if batch_norm:
            model.add(BatchNormalization())
        if dropout > 0:
            model.add(Dropout(dropout))
    model.add(GRU(num_units, return_sequences=False, activation=activation, kernel_regularizer=regularizers.l2(l2_reg)))
    if batch_norm:
        model.add(BatchNormalization())
    if dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation='softmax'))
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Compile model

def compile_model(model, learning_rate):
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Train model

def train_model(model, X_train, y_train, X_val, y_val, batch_size, epochs, verbose):
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), verbose=verbose)
    return history

# Evaluate model

def evaluate_model(model, X_test, y_test, batch_size, verbose):
    score = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=verbose)
    return score

# Predict model

def predict_model(model, X_test, batch_size, verbose):
    y_pred = model.predict(X_test, batch_size=batch_size, verbose=verbose)
    return y_pred

# Save model

def save_model(model, model_name):
    model.save(model_name)

# Load model

def load_model(model_name):
    model = load_model(model_name)
    return model

# Save history

def save_history(history, history_name):
    with open(history_name, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

# Load history

def load_history(history_name):
    with open(history_name, 'rb') as file_pi:
        history = pickle.load(file_pi)
    return history

# Plot history


# Speed up training

def speed_up_training(model, train_generator, validation_generator, epochs, batch_size, history_name):
    history = model.fit_generator(train_generator, steps_per_epoch=train_generator.samples // batch_size, epochs=epochs, validation_data=validation_generator, validation_steps=validation_generator.samples // batch_size, verbose=1)
    plot_learning_curve(history.history, history_name)
    return history
