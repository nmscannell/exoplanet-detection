from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from tensorflow.python.keras.callbacks import TerminateOnNaN
from tensorflow.python.keras.regularizers import l1, l2
from process_data import build_sets, build_astro_sets
from LSTM import build_lstm, build_cnn_lstm, build_sep_cnn_lstm, mult_in_lstm
import logging as log
import math, pickle, time, os

len_local = 201
len_global = 2001
n_features = 1


def f_opt(params):
    train, val, test = build_astro_sets(params['batch_size'])
    model = mult_in_lstm(len_local, len_global, n_features, params['lr'], params['n_layers'],
			 params['loc_n1'], params['loc_n2'], params['glob_n1'], params['glob_n2'], 
			 params['n1'], params['n2'], params['n3'], params['drop1'], params['drop2'])
    log.info(f'model: {model.summary()}')
    history = model.fit(train,
                        shuffle=True,
                        validation_data=val,
                        epochs=params['epochs'],
                        callbacks=[TerminateOnNaN()],
                        verbose=1)
    loss = min(history.history['val_loss'])
    if math.isnan(loss):
        loss = float('inf')
    log.info(f'lr: {params["lr"]}')
    log.info(f'loc_n1: {params["loc_n1"]}')
    log.info(f'glob_n1: {params["glob_n1"]}')
    log.info(f'n1: {params["n1"]}')
    log.info(f'n2: {params["n2"]}')
    log.info(f'n3: {params["n3"]}')
    log.info(f'drop1: {params["drop1"]}')
    log.info(f'drop2: {params["drop2"]}')
    log.info(f'loss: {loss}')
    return {'loss': loss, 'status': STATUS_OK}


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
log.basicConfig(filename='lstm_opt.log', level=log.INFO)
space = {
    'batch_size': 128,
    'epochs': 150,
    'lr': hp.uniform('lr', 0.002, 0.01),
    'loc_n1': hp.choice('loc1', [64, 128, 256]),
    'loc_n2': 0,
    'glob_n1': hp.choice('glob1', [64, 128, 256]),
    'glob_n2': 0,
    'n1': hp.choice('n1', [64, 128]),
    'n2': hp.choice('n2', [32, 64]),
    'n3': hp.choice('n3', [16, 32]),
    'n_layers': 1,
    'drop1': hp.uniform('drop1', 0, 1),
    'drop2': hp.uniform('drop2', 0, 1)
}

log.info('\n\noptimizing lstm')
log.info(f'current time: {time.ctime()}')
trials = Trials()
best = fmin(f_opt, space, algo=tpe.suggest, max_evals=75, trials=trials)
log.info(f'space: {space}')
log.info(f'best: {best}')
with open('lstm_res/multi_best_params.pkl', 'wb') as f:
    pickle.dump(best, f)

