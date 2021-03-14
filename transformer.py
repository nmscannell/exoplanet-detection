from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Flatten, MultiHeadAttention, Dropout, Conv1D, MaxPooling1D, Input, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.regularizers import l1, l2
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, AUC, TrueNegatives, TruePositives, FalseNegatives, FalsePositives

metrics = [BinaryAccuracy(), Precision(), Recall(), AUC(), TrueNegatives(), TruePositives(), FalseNegatives(), FalsePositives()]


def basic_trans(local_steps, global_steps, n_features, lr,
                 n1, n2, n3, drop1, drop2, num_heads, key_dim, reg):
    inputA = Input(shape=(local_steps, n_features))
    inputB = Input(shape=(global_steps, n_features))
    out = MultiHeadAttention(num_heads, key_dim, kernel_regularizer=l2(reg))(inputA, inputB)
    out = Flatten()(out)
#    out = Dense(n1, activation='relu')(out)
#    out = Dropout(drop1)(out)
#    out = Dense(n2, activation='relu')(out)
#    out = Dropout(drop2)(out)
#    out = Dense(n3, activation='relu')(out)
    out = Dense(1, activation='sigmoid')(out)
    model = Model(inputs=[inputA, inputB], outputs=out)
    opt = Adam(lr=lr)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=metrics)
    return model


def basic_trans2(local_steps, global_steps, n_features, lr,
                 n1, n2, n3, drop1, drop2, num_heads, key_dim, reg):
    inputA = Input(shape=(local_steps, n_features))
    inputB = Input(shape=(global_steps, n_features))
    loc = MultiHeadAttention(num_heads, key_dim, kernel_regularizer=l2(reg))(inputA, inputA)
    loc = Flatten()(loc)
    glob = MultiHeadAttention(num_heads, key_dim, kernel_regularizer=l2(reg))(inputB, inputB)
    glob = Flatten()(glob)
    out = concatenate([loc, glob])
    out = Dense(n1, activation='relu')(out)
    out = Dropout(drop1)(out)
    out = Dense(n2, activation='relu')(out)
    out = Dropout(drop2)(out)
    out = Dense(n3, activation='relu')(out)
    out = Dense(1, activation='sigmoid')(out)
    model = Model(inputs=[inputA, inputB], outputs=out)
    opt = Adam(lr=lr)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=metrics)
    return model


def cnn_trans(local_steps, global_steps, n_features, lr, loc_f1, loc_k1, loc_p1,
                   glob_f1, glob_k1, glob_p1, glob_f2, glob_k2, glob_p2,
                   n_neurons1, n_neurons2, n_neurons3, drop1, drop2, num_heads, key_dim):
    inputA = Input(shape=(local_steps, n_features))
    inputB = Input(shape=(global_steps, n_features))
    local = Conv1D(loc_f1, loc_k1, padding='same', activation='relu')(inputA)
    local = Conv1D(loc_f1, loc_k1, padding='same', activation='relu')(local)
    local = MaxPooling1D(loc_p1)(local)

    glob = Conv1D(glob_f1, glob_k1, padding='same', activation='relu')(inputB)
    glob = Conv1D(glob_f1, glob_k1, padding='same', activation='relu')(glob)
    glob = MaxPooling1D(glob_p1)(glob)
    glob = Conv1D(glob_f2, glob_k2, padding='same', activation='relu')(glob)
    glob = MaxPooling1D(glob_p2)(glob)

    joined = MultiHeadAttention(num_heads, key_dim, kernel_regularizer='l2')(local, glob)
    joined = Flatten()(joined)
    joined = Dense(n_neurons1, activation='relu', kernel_regularizer='l2')(joined)
    joined = Dropout(drop1)(joined)
    joined = Dense(n_neurons2, activation='relu', kernel_regularizer=None)(joined)
    joined = Dropout(drop2)(joined)
    joined = Dense(n_neurons3, activation='relu', kernel_regularizer=None)(joined)
    out = Dense(1, activation='sigmoid')(joined)
    model = Model(inputs=[inputA, inputB], outputs=out)
    opt = Adam(lr=lr)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=metrics)
    return model


def cnn_trans2(local_steps, global_steps, n_features, lr, loc_f1, loc_k1, loc_p1,
              glob_f1, glob_k1, glob_p1, glob_f2, glob_k2, glob_p2,
              n_neurons1, n_neurons2, n_neurons3, drop1, drop2,
              num_heads1, key_dim1, num_heads2, key_dim2, num_heads3, key_dim3):
    inputA = Input(shape=(local_steps, n_features))
    inputB = Input(shape=(global_steps, n_features))
    local = Conv1D(loc_f1, loc_k1, padding='same', activation='relu')(inputA)
    local = Conv1D(loc_f1, loc_k1, padding='same', activation='relu')(local)
    local = MaxPooling1D(loc_p1)(local)
    local = MultiHeadAttention(num_heads1, key_dim1, kernel_regularizer='l2')(local, local)
    local = Flatten()(local)

    glob = Conv1D(glob_f1, glob_k1, padding='same', activation='relu')(inputB)
    glob = Conv1D(glob_f1, glob_k1, padding='same', activation='relu')(glob)
    glob = MaxPooling1D(glob_p1)(glob)
    glob = Conv1D(glob_f2, glob_k2, padding='same', activation='relu')(glob)
    glob = MaxPooling1D(glob_p2)(glob)
    glob = MultiHeadAttention(num_heads2, key_dim2, kernel_regularizer='l2')(glob, glob)
    glob = Flatten()(glob)

    joined = concatenate([local, glob])
    joined = Dense(n_neurons1, activation='relu', kernel_regularizer=None)(joined)
    joined = Dropout(drop1)(joined)
    joined = Dense(n_neurons2, activation='relu', kernel_regularizer=None)(joined)
    joined = Dropout(drop2)(joined)
    joined = Dense(n_neurons3, activation='relu', kernel_regularizer=None)(joined)
    out = Dense(1, activation='sigmoid')(joined)
    model = Model(inputs=[inputA, inputB], outputs=out)
    opt = Adam(lr=lr)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=metrics)
    return model

