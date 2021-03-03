from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, concatenate, Input
from tcn import TCN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, AUC, TrueNegatives, TruePositives, FalseNegatives, FalsePositives

metrics = [BinaryAccuracy(), Precision(), Recall(), AUC(), TrueNegatives(), TruePositives(), FalseNegatives(), FalsePositives()]


def build_cnn_model(n_steps, n_features, lr, n_filters, k_size, p_size, n_neurons1, n_neurons2, n_neurons3, drop1, reg='None'):
    model = Sequential()
    model.add(Conv1D(n_filters, k_size, input_shape=(n_steps, n_features), activation='relu', padding='same'))
    model.add(MaxPooling1D(p_size))  # default stride is 1
    model.add(Dropout(drop1))
    model.add(Flatten())
    model.add(Dense(n_neurons1, activation='relu', kernel_regularizer=reg))
    # model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=reg))
    opt = Adam(lr=lr)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics='val_acc')
    return model


def build_astronet(len_local, len_global, n_features, drop1, lr):
    inputA = Input(shape=(len_local, n_features))
    inputB = Input(shape=(len_global, n_features))

    loc = Conv1D(16, 5, activation='relu')(inputA)
    loc = Conv1D(16, 5, activation='relu')(loc)
    loc = MaxPooling1D(7, strides=2)(loc)
    loc = Conv1D(32, 5, activation='relu')(loc)
    loc = Conv1D(32, 5, activation='relu')(loc)
    loc = MaxPooling1D(7, strides=2)(loc)
    loc = Flatten()(loc)
    glob = Conv1D(16, 5, activation='relu')(inputB)
    glob = Conv1D(16, 5, activation='relu')(glob)
    glob = MaxPooling1D(5, strides=2)(glob)
    glob = Conv1D(32, 5, activation='relu')(glob)
    glob = Conv1D(32, 5, activation='relu')(glob)
    glob = MaxPooling1D(5, strides=2)(glob)
    glob = Conv1D(64, 5, activation='relu')(glob)
    glob = Conv1D(64, 5, activation='relu')(glob)
    glob = MaxPooling1D(5, strides=2)(glob)
    glob = Conv1D(128, 5, activation='relu')(glob)
    glob = Conv1D(128, 5, activation='relu')(glob)
    glob = MaxPooling1D(5, strides=2)(glob)
    glob = Conv1D(256, 5, activation='relu')(glob)
    glob = Conv1D(256, 5, activation='relu')(glob)
    glob = MaxPooling1D(5, strides=2)(glob)
    glob = Flatten()(glob)
    joined = concatenate([loc, glob])
    joined = Dense(512, activation='relu')(joined)
    joined = Dropout(drop1)(joined)
    joined = Dense(512, activation='relu')(joined)
    joined = Dropout(drop1)(joined)
    joined = Dense(512, activation='relu')(joined)
    joined = Dropout(drop1)(joined)
    joined = Dense(512, activation='relu')(joined)
    joined = Dropout(drop1)(joined)
    out = Dense(1, activation='sigmoid')(joined)
    model = Model(inputs=[inputA, inputB], outputs=out)
    opt = Adam(lr=lr, epsilon=(10**(-8)))
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=metrics)
    return model


def build_astronet2(len_local, len_global, n_features, drop1, lr):
    inputA = Input(shape=(len_local, n_features))
    inputB = Input(shape=(len_global, n_features))

    loc = Conv1D(16, 5, activation='relu', padding='same')(inputA)
    loc = Conv1D(16, 5, activation='relu', padding='same')(loc)
    loc = MaxPooling1D(5, strides=2)(loc)
    loc = Conv1D(32, 5, activation='relu', padding='same')(loc)
    loc = Conv1D(32, 5, activation='relu', padding='same')(loc)
    loc = MaxPooling1D(5, strides=2)(loc)
    loc = Flatten()(loc)
    
    glob = Conv1D(16, 5, activation='relu', padding='same')(inputB)
    glob = Conv1D(16, 5, activation='relu', padding='same')(glob)
    glob = MaxPooling1D(5, strides=2)(glob)
    glob = Conv1D(32, 5, activation='relu', padding='same')(glob)
    glob = Conv1D(32, 5, activation='relu', padding='same')(glob)
    glob = MaxPooling1D(5, strides=2)(glob)
    glob = Conv1D(64, 5, activation='relu', padding='same')(glob)
    glob = Conv1D(64, 5, activation='relu', padding='same')(glob)
    glob = MaxPooling1D(5, strides=2)(glob)
    glob = Flatten()(glob)
    joined = concatenate([loc, glob])
    joined = Dense(64, activation='relu')(joined)
    joined = Dropout(drop1)(joined)
    joined = Dense(32, activation='relu')(joined)
    out = Dense(1, activation='sigmoid')(joined)
    model = Model(inputs=[inputA, inputB], outputs=out)
    opt = Adam(lr=lr, epsilon=(10**(-8)))
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=metrics)
    return model


def build_tcn(len_local, len_global, n_features, drop1, lr):
    inputA = Input(shape=(len_local, n_features))
    inputB = Input(shape=(len_global, n_features))

    loc = TCN(nb_filters=32)(inputA)
    loc = TCN(nb_filters=32)(loc)
    loc = MaxPooling1D(5, strides=2)(loc)
    loc = TCN(return_sequences=False)(loc)
#    loc = Conv1D(32, 5, activation='relu', padding='same')(loc)
#    loc = Conv1D(32, 5, activation='relu', padding='same')(loc)
#    loc = MaxPooling1D(5, strides=2)(loc)
#    loc = Flatten()(loc)
    
    glob = TCN()(inputB)
    glob = TCN()(glob)
    glob = MaxPooling1D(5, strides=2)(glob)
    glob = TCN(nb_filters=128, return_sequences=False)(glob)
#    glob = Conv1D(16, 5, activation='relu', padding='same')(glob)
#    glob = MaxPooling1D(5, strides=2)(glob)
#    glob = Conv1D(32, 5, activation='relu', padding='same')(glob)
#    glob = Conv1D(32, 5, activation='relu', padding='same')(glob)
#    glob = MaxPooling1D(5, strides=2)(glob)
#    glob = Conv1D(64, 5, activation='relu', padding='same')(glob)
#    glob = Conv1D(64, 5, activation='relu', padding='same')(glob)
#    glob = MaxPooling1D(5, strides=2)(glob)
#    glob = Flatten()(glob)
    joined = concatenate([loc, glob])
    joined = Dense(64, activation='relu')(joined)
    joined = Dropout(drop1)(joined)
    joined = Dense(32, activation='relu')(joined)
    out = Dense(1, activation='sigmoid')(joined)
    model = Model(inputs=[inputA, inputB], outputs=out)
    opt = Adam(lr=lr, epsilon=(10**(-8)))
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=metrics)
    return model

