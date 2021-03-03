from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, MultiHeadAttention, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten, Input, concatenate, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.regularizers import l1, l2
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, AUC, TrueNegatives, TruePositives, FalseNegatives, FalsePositives

metrics = [BinaryAccuracy(), Precision(), Recall(), AUC(), TrueNegatives(), TruePositives(), FalseNegatives(), FalsePositives()]


def mult_in_lstm(local_steps, global_steps, n_features, lr, n_layers, loc_n1, loc_n2, glob_n1, glob_n2,
                 n1, n2, n3, drop1, drop2):
    inputA = Input(shape=(local_steps, n_features))
    inputB = Input(shape=(global_steps, n_features))
    local = LSTM(loc_n1, kernel_regularizer='l2', return_sequences=False)(inputA)
    #local = LSTM(n_neurons2)
    #local = Dropout(drop1)(local)
    glob = LSTM(glob_n1, kernel_regularizer='l2', return_sequences=False)(inputB)
    #glob = LSTM(glob_n2, kernel_regularizer='l2')(glob)
    #glob = Dropout(drop2)(glob)
    joined = concatenate([local, glob])
    joined = Dense(n1, activation='relu')(joined)
    joined = Dropout(drop1)(joined)
    joined = Dense(n2, activation='relu')(joined)
    joined = Dense(n3, activation='relu')(joined)
    joined = Dropout(drop2)(joined)
    out = Dense(1, activation='sigmoid')(joined)
    model = Model(inputs=[inputA, inputB], outputs=out)
    opt = Adam(lr=lr)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=metrics)
    return model


def mult_in_lstm_attn(local_steps, global_steps, n_features, lr, n_layers, loc_n1, loc_n2, glob_n1, glob_n2,
                 n1, n2, n3, drop1, drop2):
    inputA = Input(shape=(local_steps, n_features))
    inputB = Input(shape=(global_steps, n_features))
    local = LSTM(loc_n1, kernel_regularizer='l2', return_sequences=False)(inputA)
    #local = LSTM(n_neurons2)
    #local = Dropout(drop1)(local)
    glob = LSTM(glob_n1, kernel_regularizer='l2', return_sequences=False)(inputB)
    #glob = LSTM(glob_n2, kernel_regularizer='l2')(glob)
    #glob = Dropout(drop2)(glob)
    joined = Attention()([local, glob])
    joined = Dense(n1, activation='relu')(joined)
    joined = Dropout(drop1)(joined)
    joined = Dense(n2, activation='relu')(joined)
    joined = Dense(n3, activation='relu')(joined)
    joined = Dropout(drop2)(joined)
    out = Dense(1, activation='sigmoid')(joined)
    model = Model(inputs=[inputA, inputB], outputs=out)
    opt = Adam(lr=lr)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=metrics)
    return model


def build_lstm(n_steps, n_features, lr, n_layers, n_neurons1, n_neurons2, n_neurons3, n_neurons4, drop1, drop2, reg):
    model = Sequential()
    if n_layers == 1:
        model.add(LSTM(n_neurons1, input_shape=(n_steps, n_features), kernel_regularizer=reg))
    else:
        model.add(LSTM(n_neurons1, input_shape=(n_steps, n_features), return_sequences=True, kernel_regularizer=reg))
        model.add(Dropout(drop1))
        model.add(LSTM(n_neurons2, kernel_regularizer=reg))
    model.add(Dropout(drop2))
    model.add(Dense(n_neurons3, activation='relu', kernel_regularizer=reg))
    model.add(Dense(n_neurons4, activation='relu', kernel_regularizer=reg))
    model.add(Dense(1, activation='sigmoid'))
    opt = Adam(lr=lr)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=metrics)
    return model


def build_cnn_lstm(local_steps, global_steps, n_features, lr, loc_f1, loc_k1, loc_p1, loc_drop1,
                   glob_f1, glob_k1, glob_p1, glob_f2, glob_k2, glob_p2, glob_drop1,
                   n_neurons1, n_neurons2, n_neurons3, n_neurons4, n_neurons5, drop1, drop2, reg):
    inputA = Input(shape=(local_steps, n_features))
    inputB = Input(shape=(global_steps, n_features))
    local = Conv1D(loc_f1, loc_k1, padding='same', activation='relu')(inputA)
    local = Conv1D(loc_f1, loc_k1, padding='same', activation='relu')(local)
    local = MaxPooling1D(loc_p1, strides=2)(local)
    local = LSTM(n_neurons1, kernel_regularizer=l2(reg))(local)
    glob = Conv1D(glob_f1, glob_k1, padding='same', activation='relu')(inputB)
    glob = Conv1D(glob_f1, glob_k1, padding='same', activation='relu')(glob)
    glob = MaxPooling1D(glob_p1, strides=2)(glob)
    glob = Conv1D(glob_f2, glob_k2, padding='same', activation='relu')(glob)
    glob = Conv1D(glob_f2, glob_k2, padding='same', activation='relu')(glob)
    glob = MaxPooling1D(glob_p2, strides=2)(glob)
    glob = LSTM(n_neurons2, kernel_regularizer=l2(reg))(glob)
    joined = concatenate([local, glob])
    joined = Dense(n_neurons3, activation='relu', kernel_regularizer=None)(joined)
    joined = Dropout(drop1)(joined)
    joined = Dense(n_neurons4, activation='relu', kernel_regularizer=None)(joined)
    joined = Dropout(drop2)(joined)
    #joined = Dense(n_neurons5, activation='relu', kernel_regularizer=None)(joined)
    out = Dense(1, activation='sigmoid')(joined)
    model = Model(inputs=[inputA, inputB], outputs=out)
    opt = Adam(lr=lr)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=metrics)
    return model


def build_cnn_lstm_attn(local_steps, global_steps, n_features, lr, loc_f1, loc_k1, loc_p1, loc_drop1,
                   glob_f1, glob_k1, glob_p1, glob_f2, glob_k2, glob_p2, glob_drop1,
                   n_neurons1, n_neurons2, n_neurons3, n_neurons4, n_neurons5, drop1, drop2):
    inputA = Input(shape=(local_steps, n_features))
    inputB = Input(shape=(global_steps, n_features))
    local = Conv1D(loc_f1, loc_k1, padding='same', activation='relu')(inputA)
    local = Conv1D(loc_f1, loc_k1, padding='same', activation='relu')(local)
    local = MaxPooling1D(loc_p1)(local)
    local = LSTM(n_neurons1, dropout=loc_drop1, kernel_regularizer='l2')(local)

    glob = Conv1D(glob_f1, glob_k1, padding='same', activation='relu')(inputB)
    glob = Conv1D(glob_f1, glob_k1, padding='same', activation='relu')(glob)
    glob = MaxPooling1D(glob_p1)(glob)
    glob = Conv1D(glob_f2, glob_k2, padding='same', activation='relu')(glob)
    glob = MaxPooling1D(glob_p2)(glob)
    glob = LSTM(n_neurons2, dropout=glob_drop1, kernel_regularizer='l2')(glob)
    joined = Attention()([local, glob])
    joined = Dense(n_neurons3, activation='relu', kernel_regularizer=None)(joined)
    joined = Dropout(drop1)(joined)
    joined = Dense(n_neurons4, activation='relu', kernel_regularizer=None)(joined)
    joined = Dropout(drop2)(joined)
    joined = Dense(n_neurons5, activation='relu', kernel_regularizer=None)(joined)
    out = Dense(1, activation='sigmoid')(joined)
    model = Model(inputs=[inputA, inputB], outputs=out)
    opt = Adam(lr=lr)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=metrics)
    return model


def build_cnn_lstm_attn2(local_steps, global_steps, n_features, lr, loc_f1, loc_k1, loc_p1, loc_drop1,
                   glob_f1, glob_k1, glob_p1, glob_f2, glob_k2, glob_p2, glob_drop1,
                   n_neurons1, n_neurons2, n_neurons3, n_neurons4, n_neurons5, drop1, drop2):
    inputA = Input(shape=(local_steps, n_features))
    inputB = Input(shape=(global_steps, n_features))
    local = Conv1D(loc_f1, loc_k1, padding='same', activation='relu')(inputA)
    local = Conv1D(loc_f1, loc_k1, padding='same', activation='relu')(local)
    local = MaxPooling1D(loc_p1)(local)

    glob = Conv1D(glob_f1, glob_k1, padding='same', activation='relu')(inputB)
    glob = Conv1D(glob_f1, glob_k1, padding='same', activation='relu')(glob)
    glob = MaxPooling1D(glob_p1)(glob)
    glob = Conv1D(glob_f2, glob_k2, padding='same', activation='relu')(glob)
    glob = Conv1D(glob_f2, glob_k2, padding='same', activation='relu')(glob)
    glob = MaxPooling1D(glob_p2)(glob)

    joined = Attention()([local, glob])
    #joined = LSTM(n_neurons1, kernel_regularizer='l2', return_sequences=True)(joined)
    joined = LSTM(n_neurons2, kernel_regularizer='l2')(joined)
    joined = Dense(n_neurons3, activation='relu')(joined)
    joined = Dropout(drop1)(joined)
    joined = Dense(n_neurons4, activation='relu', kernel_regularizer=None)(joined)
    joined = Dropout(drop2)(joined)
    joined = Dense(n_neurons5, activation='relu', kernel_regularizer=None)(joined)
    out = Dense(1, activation='sigmoid')(joined)
    model = Model(inputs=[inputA, inputB], outputs=out)
    opt = Adam(lr=lr)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=metrics)
    return model


def build_sep_cnn_lstm(n_steps, n_features, lr, n_filters, n_filters2, k_size, p_size,
                       n_neurons1, n_neurons2, n_neurons3, drop1, drop2, reg='None'):
    inp = Input(shape=(n_steps, n_features))
    rnn = LSTM(n_neurons1, return_sequences=True, kernel_regularizer=reg)(inp)
    rnn = LSTM(n_neurons2)(rnn)
    rnn = Dropout(drop1)(rnn)
    cnn = Conv1D(n_filters, k_size, padding='same', activation='relu')(inp)
    cnn = MaxPooling1D(p_size)(cnn)
    cnn = Conv1D(n_filters2, k_size, padding='same', activation='relu', kernel_regularizer=reg)(cnn)
    cnn = MaxPooling1D(p_size)(cnn)
    cnn = Flatten()(cnn)
    cnn = Dropout(drop2)(cnn)
    rnn = concatenate([rnn, cnn])
    rnn = Dense(n_neurons3, activation='relu', kernel_regularizer=reg)(rnn)
    out = Dense(1, activation='sigmoid')(rnn)
    model = Model(inp, out)
    opt = Adam(lr=lr)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=metrics)
    return model
