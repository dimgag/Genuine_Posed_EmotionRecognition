# 
# Multi-task Multi-label classification model for emotion recognition in images

def build_model(input_shape, output_shape, num_layers, num_units, dropout, learning_rate, l2_reg, activation, batch_norm, batch_size, num_classes):
    model = Sequential()
    model.add(Conv2D(num_units, (3, 3), input_shape=input_shape, activation=activation, kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(Conv2D(num_units, (3, 3), activation=activation, kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))
    model.add(Conv2D(num_units, (3, 3), activation=activation, kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(Conv2D(num_units, (3, 3), activation=activation, kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(num_units, activation=activation, kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation='sigmoid'))
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Compile model

def compile_model(model, learning_rate):
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# Define MFL loss function

def mfl_loss(y_true, y_pred):
    loss = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
    return loss

# Define MFL accuracy function

def mfl_accuracy(y_true, y_pred):
    accuracy = K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)
    return accuracy


# compile model

def compile_model(model, learning_rate):
    optimizer = Adam(lr=learning_rate)
    model.compile(loss=mfl_loss, optimizer=optimizer, metrics=[mfl_accuracy])
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

