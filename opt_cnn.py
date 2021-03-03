from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from tensorflow.python.keras.callbacks import TerminateOnNaN
from tensorflow.python.keras.regularizers import l1, l2
from process_data import build_sets, build_astro_sets
from LSTM import build_lstm, build_cnn_lstm, build_sep_cnn_lstm, mult_in_lstm
from temporal_cnn import build_astronet, build_astronet2
import logging as log
import math, pickle, time, os

len_local = 201
len_global = 2001
n_features = 1


def f_opt(params):
    train, val, test = build_astro_sets(params['batch_size'])
#    model = build_astronet(len_local, len_global, n_features, params['drop1'], params['lr'])
    model = build_astronet2(len_local, len_global, n_features, params['drop1'], params['lr'])
    log.info(f'model: {model.summary()}')
    log.info(f'lr: {params["lr"]}')
    log.info(f'drop1: {params["drop1"]}')
    log.info(f'bs: {params["batch_size"]}')
    history = model.fit(train,
                        shuffle=True,
                        validation_data=val,
                        epochs=params['epochs'],
                        callbacks=[TerminateOnNaN()],
                        verbose=1)
    loss = min(history.history['val_loss'])
    if math.isnan(loss):
        loss = float('inf')
    log.info(f'loss: {loss}')
    return {'loss': loss, 'status': STATUS_OK}


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
log.basicConfig(filename='astro2_opt2.log', level=log.INFO)
space = {
    'batch_size': hp.choice('bs', [32, 64, 128]),
    'epochs': 50,
    'lr': hp.uniform('lr', 0.00005, 0.01),
    'drop1': hp.uniform('drop1', 0, 1),
}

log.info('\n\noptimizing astronet2')
log.info(f'current time: {time.ctime()}')
trials = Trials()
best = fmin(f_opt, space, algo=tpe.suggest, max_evals=50, trials=trials)
#log.info(f'space: {space}')
#log.info(f'best: {best}')
with open('smol_astro_best_params2.pkl', 'wb') as f:
    pickle.dump(best, f)

