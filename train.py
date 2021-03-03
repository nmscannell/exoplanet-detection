from tensorflow.python.keras.callbacks import ModelCheckpoint, TerminateOnNaN
from tensorflow.python.keras.models import load_model
from process_data import build_sets, build_astro_sets
from LSTM import build_lstm, build_cnn_lstm, build_sep_cnn_lstm, mult_in_lstm, build_cnn_lstm_attn, build_cnn_lstm_attn2
from transformer import basic_trans, cnn_trans, cnn_trans2, basic_trans2
from temporal_cnn import build_astronet, build_astronet2, build_tcn
import pickle, os, json
import logging as log
import matplotlib.pyplot as plt

log.basicConfig(filename='train_astro.log', level=log.INFO)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
with open('smol_astro_best_params2.pkl', 'rb') as f:
    params = pickle.load(f)
#with open('params.json', 'r') as f:
#    params = json.load(f)

len_local = 201
len_global = 2001
n_features = 1
model_n = 'smol_astro_opt'
batch_size = params['bs']
log.info(f'\n\nmodel name: {model_n}')
log.info(f'batch size: {batch_size}')
train, val, test = build_astro_sets(batch_size)

if not os.path.exists(model_n):
    log.info('building and training model')
#    log.info(f'learning rate: {params["lr"]}')
    '''
    log.info(f'n_layers: {params["n_layers"]}')
    log.info(f'loc_f1: {params["loc_f1"]}')
    log.info(f'loc_k1: {params["loc_k1"]}')
    log.info(f'loc_p1: {params["loc_p1"]}')
    log.info(f'glob_f1: {params["glob_f1"]}')
    log.info(f'glob_k1: {params["glob_k1"]}')
    log.info(f'glob_p1: {params["glob_p1"]}')
    log.info(f'glob_f2: {params["glob_f2"]}')
    log.info(f'glob_k2: {params["glob_k2"]}')
    log.info(f'glob_p2: {params["glob_p2"]}')
    log.info(f'n1: {params["n1"]}')
    log.info(f'n2: {params["n2"]}')
    log.info(f'n3: {params["n3"]}')
    log.info(f'n4: {params["n4"]}')
    log.info(f'n5: {params["n5"]}')
    log.info(f'drop1: {params["drop1"]}')
    log.info(f'drop2: {params["drop2"]}')
    log.info(f'num_heads: {params["num_heads"]}')
    log.info(f'key_dim: {params["key_dim"]}')
'''
    model = build_astronet2(len_local, len_global, n_features, params['drop1'], params['lr'])
#    model = build_tcn(len_local, len_global, n_features, 0.4, 0.001)
#    model = build_cnn_lstm(len_local, len_global, n_features, params['lr'], params['loc_f1'], params['loc_k1'],
#                                params['loc_p1'], params['loc_drop1'], params['glob_f1'], params['glob_k1'],
#                                params['glob_p1'], params['glob_f2'], params['glob_k2'], params['glob_p2'],
#                                params['glob_drop1'], params['n1'], params['n2'], params['n3'], params['n4'],
#                                params['n5'], params['drop1'], params['drop2'], 0.001)
#    model = mult_in_lstm(len_local, len_global, n_features, params['lr'], params['n_layers'], params['n2'], 0,
#                         params['n1'], params['n2'], params['n3'], params['n4'], params['n5'], params['drop1'],
#                         params['drop2'])
#    model = basic_trans(len_local, len_global, n_features, params['lr'], params['n3'], params['n4'], params['n5'],
#                        params['drop1'], params['drop2'], params['num_heads'], params['key_dim'])
#    model = basic_trans2(len_local, len_global, n_features, params['lr'], params['n3'], params['n4'], params['n5'],
#                        params['drop1'], params['drop2'], params['num_heads'], params['key_dim'])
#    model = cnn_trans(len_local, len_global, n_features, params['lr'], params['loc_f1'], params['loc_k1'],
#                      params['loc_p1'], params['glob_f1'], params['glob_k1'], params['glob_p1'],
#                      params['glob_f2'], params['glob_k2'], params['glob_p2'], params['n1'],
#                      params['n2'], params['n3'], params['drop1'], params['drop2'],
#                      params['num_heads'], params['key_dim'])
#    model = cnn_trans2(len_local, len_global, n_features, params['lr'], params['loc_f1'], params['loc_k1'],
#                      params['loc_p1'], params['glob_f1'], params['glob_k1'], params['glob_p1'],
#                      params['glob_f2'], params['glob_k2'], params['glob_p2'], params['n1'],
#                      params['n2'], params['n3'], params['drop1'], params['drop2'],
#                      params['num_heads1'], params['key_dim1'], params['num_heads2'], params['key_dim2'],
#                      params['num_heads3'], params['key_dim3'])
    model.summary()
    model.summary(print_fn=log.info)
    modelCheckpoint = ModelCheckpoint(model_n,
                                      monitor='val_binary_accuracy',
                                      save_best_only=True,
                                      mode='max',
                                      verbose=1,
                                      save_weights_only=False)
    history = model.fit(train,
                        shuffle=True,
                        validation_data=val,
                        epochs=75,
                        callbacks=[modelCheckpoint, TerminateOnNaN()],
                        verbose=1)
    plt.figure()
    plt.title('Loss', loc='center')
    plt.plot(history.history['loss'], 'b', label='train')
    plt.plot(history.history['val_loss'], 'r', label='validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('figures/'+model_n+'_loss.png')
    plt.clf()
    plt.plot(history.history['binary_accuracy'], 'b', label='train')
    plt.plot(history.history['val_binary_accuracy'], 'r', label='validation')
    plt.title('Accuracy', loc='center')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('figures/'+model_n+'_accuracy.png')

log.info('loading best trained model')
model = load_model(model_n)
result = model.evaluate(test)
log.info(f'test loss and accuracy: {result}')
log.info(f'metric labels: {model.metrics_names}')
print(result)
print(model.metrics_names)
