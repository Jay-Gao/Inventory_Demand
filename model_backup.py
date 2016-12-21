#! ~/anaconda3/bin/python3
# -*- coding: utf-8 -*-

import os
import pickle
import logging

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split

import preprocess as pp


# Configure logging
logger = logging.getLogger('model')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


DTYPES = {'Agencia_ID': 'int32', 'Canal_ID': 'int32', 'Ruta_SAK': 'int32', 'Cliente_ID': 'int32', 'Producto_ID': 'int32',
          'Demanda_uni_equil': 'float32', 'client_count': 'int32', 'lag1': 'float32', 'lag2': 'float32', 'lag3': 'float32',
          'lag4': 'float32', 'sum_of_lag_features': 'float32', 'sum_of_prior_demand': 'float32','mean_of_prior_demand': 'float32',
          'product_client_pairs_freq': 'int32', 'price': 'float32', 'weight': 'float32', 'pieces': 'int32',
          'weight_per_piece': 'float32', 'brand': 'object'}

FEATURES = ['client_count', 'lag1', 'lag2', 'lag3', 'lag4', 'sum_of_lag_features', 'sum_of_prior_demand',
             'mean_of_prior_demand', 'product_client_pairs_freq', 'price', 'weight', 'pieces', 'weight_per_piece']

# Semana,Agencia_ID,Canal_ID,Ruta_SAK,Cliente_ID,Producto_ID

# def mean_value_model(train_set, train_dict, **kwargs):
#     """
#     return mean value of a few month as the predicted demand.
#     Args:
#         train_set: training data set.
#         **kwargs:
#
#     Returns: predicted demand
#
#     """
# #     train_dict = {col: set(train_set[col].unique()) for col in train_set.columns}
# #     test_dict = dict()
# #     for col, value in kwargs.items():
# #         if value in train_dict[col]:
# #             test_dict[col] = value
# #     pprint(test_dict)
# #     df = select_data(dataframe=train_set, **test_dict)
#     if df is None:
#         print('selected dataframe is None.')
#         return None
#     return df['Demanda_uni_equil'].mean()


# %% mean_value moel
def get_demand(row, train_dict_5, train_dict_2, train_dict_1):

    row = row.values.tolist()
    key = tuple(row[2:])
    demand = train_dict_5.get(key, -1)
    if demand < 0:
        key = tuple(row[2: 4])
        demand = train_dict_2.get(key, -1)
        if demand < 0:
            # key = tuple(row[2])
            key = row[2]
            demand = train_dict_1.get(key, 0)
            return demand
        else:
            return demand
    else:
        return demand


def mean_model():
    dtype = {'Semana': 'int32',
             'Agencia_ID': 'int32',
             'Canal_ID': 'int32',
             'Ruta_SAK': 'int32',
             'Producto_ID': 'int32',
             'Cliente_ID': 'int32',
             'Demanda_uni_equil': 'int32'}

    CHUNK_SIZE = 100000

    # column = ['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID', 'Demanda_uni_equil']
    train_column = ['Semana', 'Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID', 'Demanda_uni_equil']
    train = pd.read_csv('raw_data/train.csv', dtype=dtype, usecols=train_column)
    print('read train.csv done.')
    # train_dict = {col: set(train_df[col].unique()) for col in train_df.columns}
    test_reader = pd.read_csv('raw_data/test.csv', dtype=dtype, chunksize=CHUNK_SIZE)
    print('read test.csv done.')

    # get dict from training set.
    if os.path.exists('data/train_dict_5.pickle'):
        with open('data/train_dict_5.pickle', 'rb') as fr:
            train_dict_5 = pickle.load(fr)
    else:
        train_dict_5 = train.groupby(['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID'])['Demanda_uni_equil'].mean().to_dict()
        with open('data/train_dict_5.pickle', 'wb') as fw:
            pickle.dump(train_dict_5, fw)
    print('get train_dict_5 done.')

    if os.path.exists('data/train_dict_2.pickle'):
        with open('data/train_dict_2.pickle', 'rb') as fr:
            train_dict_2 = pickle.load(fr)
    else:
        train_dict_2 = train.groupby(['Agencia_ID', 'Canal_ID'])['Demanda_uni_equil'].mean().to_dict()
        with open('data/train_dict_2.pickle', 'wb') as fw:
            pickle.dump(train_dict_2, fw)
    print('get train_dict_2 done.')

    if os.path.exists('data/train_dict_1.pickle'):
        with open('data/train_dict_1.pickle', 'rb') as fr:
            train_dict_1 = pickle.load(fr)
    else:
        train_dict_1 = train.groupby(['Agencia_ID'])['Demanda_uni_equil'].mean().to_dict()
        with open('data/train_dict_1.pickle', 'wb') as fw:
            pickle.dump(train_dict_1, fw)
    print('get train_dict_1 done.')

    # free train
    del train
    print('train freed')

    test = test_reader.get_chunk()
    test['Demanda_uni_equil'] = test.apply(lambda row: get_demand(row, train_dict_5, train_dict_2, train_dict_1), axis=1)
    answer = test[['id', 'Demanda_uni_equil']]
    answer['Demanda_uni_equil'] = answer['Demanda_uni_equil'].round()
    answer['Demanda_uni_equil'] = answer['Demanda_uni_equil'].apply(int)

    for test in test_reader:
        test['Demanda_uni_equil'] = test.apply(lambda row: get_demand(row, train_dict_5, train_dict_2, train_dict_1), axis=1)
        tmp = test[['id', 'Demanda_uni_equil']]
        tmp['Demanda_uni_equil'] = tmp['Demanda_uni_equil'].round()
        tmp['Demanda_uni_equil'] = tmp['Demanda_uni_equil'].apply(int)
        answer = pd.concat([answer, tmp])

    print('done with test.')

    # free dict
    del train_dict_1, train_dict_2, train_dict_5
    print('train_dict freed.')

    # answer = test[['id', 'Demanda_uni_equil']]
    # answer['Demanda_uni_equil'] = answer['Demanda_uni_equil'].round()
    answer.to_csv(path='submission/mean1_model.csv', index=False)
    return answer


#----------------------------------------------------------------------------------------------------------------------#
# Process with xgboost model.
def train_xgb_model(data=None, features=None, train_number=1, batch_size=0.1, cv_size=0.2, log_demand=True, normalize=False):

    # load data
    logger.info('Loading train data...')
    if data is None:
        data = pd.read_csv('new_data/reg_train_{:02}.csv'.format(train_number), dtype=DTYPES)

    logger.info('Setting features for training...')
    if features is None:
        features = ['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID', 'client_count',
                    'lag1', 'lag2', 'lag3', 'lag4', 'sum_of_lag_features', 'sum_of_prior_demand',
                    'mean_of_prior_demand', 'product_client_pairs_freq', 'weight', 'pieces', 'weight_per_piece']

    # sampling
    logger.info('Sampling...')
    data = data.take(np.random.permutation(len(data))[: int(batch_size * len(data))])

    # prepare training set.
    logger.info("Preparing training set...")
    if log_demand:
        data.loc[:, 'Demanda_uni_equil'] = np.log(data.Demanda_uni_equil.values+1)

    x_data = data[features]
    feature_names = x_data.columns.tolist()
    x_data = x_data.values
    if normalize:
        x_data = x_data / x_data.max(axis=0)
    y_data = data['Demanda_uni_equil'].values

    # free memory
    del data

    # make cross-validation set.
    logger.info('Cross-validation...')
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=cv_size)
    dtrain = xgb.DMatrix(data=x_train, label=y_train, feature_names=feature_names)
    dtest = xgb.DMatrix(data=x_test, label=y_test, feature_names=feature_names)

    # free memory
    del x_data, x_train, x_test, y_train, y_test

    # xgboost parameters
    params = {
        'booster': 'gbtree',
        'max_depth': 10,
        'eta': 0.3,
        'silent': 1,
        'nthread': 6,
        'subsample': 0.6,
        'colsample_bytree': 0.8,
        'lambda': 1.0,
        'min_child_weight': 3
    }
    print('parameters: \n', params)
    watchlist = [(dtrain, 'train'), (dtest, 'eval')]
    logger.info('Training...')
    xgb_reg = xgb.train(params=params, dtrain=dtrain, num_boost_round=40, evals=watchlist, early_stopping_rounds=5)
    logger.info('Done with training.')

    return xgb_reg


def predict(xgb_model_path=None, week=10, features=None, data=None, log_demand=True, normalize=False):

    bst = xgb.Booster(params={'nthread': 3})
    bst.load_model(xgb_model_path)

    # load data for prediction.
    logger.info('Loading data for prediction...')
    if data is None:
        data = pd.read_csv('new_data/test_week_{}.csv'.format(week), dtype=DTYPES)

    # prepare data.
    feature_data = data[features].values
    if normalize:
        feature_data = week / feature_data.max(axis=0)
    dtest = xgb.DMatrix(feature_data, label=None, feature_names=features)

    # free memory.
    del feature_data

    # predict data set.
    logger.info('Predicting...')
    y_test = bst.predict(dtest)

    if log_demand:
        data['Demanda_uni_equil'] = np.exp(y_test) - 1
    else:
        data['Demanda_uni_equil'] = y_test

    # save predicted results.
    data[['id', 'Demanda_uni_equil']].to_csv('submission/submission_week_{}.csv'.format(week), index=False, float_format='%.4f')

    # save week demand information for next week's feature extraction.
    data[['Semana', 'Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID', 'Demanda_uni_equil']].to_csv \
        ('new_data/train_{}.csv'.format(week), index=False, float_format='%.4f')
    logger.info('Done with prediction.')


def run(model_number, train_number):

    model_path = 'models/{:02}.model'.format(model_number)

    logger.info('Loading model...')
    if not os.path.exists(model_path):
        logger.info('Model does not exist, training model...')
        train_data = pd.read_csv('new_data/reg_train_{:02}.csv'.format(train_number), dtype=DTYPES)
        xgb_reg = train_xgb_model(data=train_data, features=FEATURES, batch_size=1.0, cv_size=0.2)
        xgb_reg.save_model(model_path)
    logger.info('Done with loading model.')

    logger.info('Predicting week 10')
    predict(xgb_model_path=model_path, week=10, features=FEATURES)
    pp.set_test_11()
    logger.info('Predicting week 11')
    predict(xgb_model_path=model_path, week=11, features=FEATURES)
    sub_10 = pd.read_csv('submission/submission_week_10.csv')
    sub_11 = pd.read_csv('submission/submission_week_11.csv')
    sub = pd.concat([sub_10, sub_11])
    sub = sub.sort_values(by='id')
    sub.to_csv('submission/submission{:02}.csv'.format(model_number), index=False)
    print('Done with submission.')


def run_based_on_week(train_number):

    model_numbers = [37, 38, 39]
    weeks = [7, 8, 9]

    logger.info('Loading total train data...')
    total_train_data = pd.read_csv('new_data/reg_train_{:02}.csv'.format(train_number), dtype=DTYPES)

    for model_number, week in zip(model_numbers, weeks):
        model_path = 'models/{:02}.model'.format(model_number)

        logger.info('Loading model{:02}...'.format(model_number))
        if not os.path.exists(model_path):
            logger.info('Model does not exist, training model...')
            # train_data = pd.read_csv('new_data/reg_train_{:02}.csv'.format(train_number), dtype=DTYPES)
            train_data = total_train_data[total_train_data.Semana == week]
            xgb_reg = train_xgb_model(data=train_data, features=FEATURES, batch_size=1.0, cv_size=0.1)
            xgb_reg.save_model(model_path)
        logger.info('Done with loading model.')

        logger.info('Predicting week 10')
        data = pd.read_csv('new_data/test_week_10_{:02}.csv'.format(train_number), dtype=DTYPES)
        predict(xgb_model_path=model_path, data=data, week=10, features=FEATURES)

        pp.set_test_11(train_number=train_number, log_demand=False)
        logger.info('Predicting week 11')
        data = pd.read_csv('new_data/test_week_11_{:02}.csv'.format(train_number), dtype=DTYPES)
        predict(xgb_model_path=model_path, data=data, week=11, features=FEATURES)
        sub_10 = pd.read_csv('submission/submission_week_10.csv')
        sub_11 = pd.read_csv('submission/submission_week_11.csv')
        sub = pd.concat([sub_10, sub_11])
        sub = sub.sort_values(by='id')
        sub.to_csv('submission/submission{:02}.csv'.format(model_number), index=False)
        print('Done with submission.')

#----------------------------------------------------------------------------------------------------------------------#
# run cluster models with xgboost.


def run_cluster_xgoost():

    # load cluster model.
    logger.info('Loading cluster model...')
    cls = pickle.load(open('models/kmeans_model', 'rb'))

    train_data = pd.read_csv('new_data/reg_train_01_cls.csv', dtype=DTYPES)
    train_data = train_data[FEATURES].values
    _, U = pp.pca_analysis(train_data)

    logger.info('Predicting week 10...')
    logger.info('PCA with week 10...')
    data_10 = pd.read_csv('new_data/test_week_10_01.csv', dtype=DTYPES)
    tmp = data_10[FEATURES].values
    tmp = (tmp - tmp.mean(axis=0)) / tmp.std(axis=0)
    pca_data_10 = np.dot(tmp, U)
    # pca_data_10 = pp.pca_analysis(data_10[FEATURES].values)
    labels_10 = cls.predict(pca_data_10)
    data_10['label'] = labels_10

    for cluster in range(0, 3):
        model_path = 'models/cluster{}.model'.format(cluster)
        logger.info('Loading cluster{}.model'.format(cluster))
        if not os.path.exists(model_path):
            logger.info('Model does not exist, training model...')
            train_data = pd.read_csv('new_data/reg_train_01_cls.csv', dtype=DTYPES)
            _train = train_data[train_data.label == cluster]
            print('There are {} training samples.'.format(len(_train)))
            xgb_reg = train_xgb_model(data=_train, features=FEATURES, batch_size=1.0, cv_size=0.2)
            xgb_reg.save_model(model_path)

        xgb_reg = xgb.Booster(params={'nthread': 6})
        xgb_reg.load_model(model_path)

        # data = pd.read_csv('new_data/test_week_10_01.csv', dtype=DTYPES)
        cls_data_10 = data_10[data_10.label == cluster]
        feature_data = cls_data_10[FEATURES].values
        dtest = xgb.DMatrix(feature_data, label=None, feature_names=FEATURES)
        del feature_data
        y_test = xgb_reg.predict(dtest)
        cls_data_10['Demanda_uni_equil'] = np.exp(y_test) - 1

        if cluster == 0:
            sub_10 = cls_data_10[['id', 'Demanda_uni_equil']]
            train_10 = cls_data_10[['Semana', 'Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID', 'Demanda_uni_equil']]
        else:
            sub_10 = pd.concat([sub_10, cls_data_10[['id', 'Demanda_uni_equil']]])
            train_10 = pd.concat([train_10, cls_data_10[['Semana', 'Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID', 'Demanda_uni_equil']]])

    logger.info('Predicting week 11...')
    use_cols = ['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Demanda_uni_equil']
    train = pp.load_data(path='raw_data/train.csv', use_cols=use_cols)
    test_set = pd.read_csv('raw_data/test.csv')
    data_11 = pp.extract_lag_features(data_frame=train, test_set=test_set, week=11)

    logger.info('PCA with week 11...')
    # pca_data_11 = pp.pca_analysis(data_11[FEATURES].values)
    tmp = data_11[FEATURES].values
    tmp = (tmp - tmp.mean(axis=0)) / (tmp.std(axis=0)+1e-5)
    pca_data_11 = np.dot(tmp, U)
    labels_11 = cls.predict(pca_data_11)
    data_11['label'] = labels_11

    for cluster in range(0, 3):
        model_path = 'models/cluster{}.model'.format(cluster)
        logger.info('Loading cluster{}.model'.format(cluster))
        xgb_reg = xgb.Booster(params={'nthread': 6})
        xgb_reg.load_model(model_path)

        # data = pd.read_csv('new_data/test_week_10_01.csv', dtype=DTYPES)
        cls_data_11 = data_11[data_11.label == cluster]
        feature_data = cls_data_11[FEATURES].values
        dtest = xgb.DMatrix(feature_data, label=None, feature_names=FEATURES)
        del feature_data
        y_test = xgb_reg.predict(dtest)
        cls_data_11['Demanda_uni_equil'] = np.exp(y_test) - 1

        if cluster == 0:
            sub_11 = cls_data_11[['id', 'Demanda_uni_equil']]
            train_11 = cls_data_11[
                ['Semana', 'Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID', 'Demanda_uni_equil']]
        else:
            sub_11 = pd.concat([sub_11, cls_data_11[['id', 'Demanda_uni_equil']]])
            train_11 = pd.concat([train_11, cls_data_11[
                ['Semana', 'Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID', 'Demanda_uni_equil']]])

    sub = pd.concat([sub_10, sub_11])
    sub = sub.sort_values(by='id')
    sub.to_csv('submission/submission_cluster.csv', index=False)
    logger.info('Done with submission.')


# current better parameters.
# smaller max_depth with larger num_boost_rounds.
# params = {
#     'booster': 'gbtree',
#     'max_depth': 10,
#     'eta': 0.2,
#     'silent': 1,
#     'nthread': 3,
#     'subsample': 0.6,
#     'lambda': 3.0,
#     'min_child_weight': 3
# }

# xgb_cv = xgb.cv(params=params, dtrain=dtrain, num_boost_round=10, nfold=5, metrics=['rmse'], show_progress=True, seed=100)

# param_grids = {'max_depth': [6, 10, 20],
#                'min_child_weight': [1, 2, 4]}
# xgb_model = xgb.XGBRegressor(learning_rate=0.1, n_estimators=2, silent=True, nthread=3, subsample=0.6, colsample_bytree=0.8)
# xgb_cv = GridSearchCV(xgb_model, param_grid=param_grids, n_jobs=3, verbose=1)
# xgb_reg.fit(x_train, y_train)

# watchlist = [(dtrain, 'train'), (dtest, 'eval')]
#----------------------------------------------------------------------------------------------------------------------#

# %%evaluation function: RMSLE( Root Mean Squared Logarithmic Error.)
#
def rmsle_loss(preds, actuals):
    """
    e = sqrt(1/N * sum_i^N(log(p_i+1) - log(a_i+1))^2)
    Args:
        preds: predicted values
        actuals: actual values
    Returns: average loss(rmsle)
    """
    log_p = np.log(preds + 1)
    log_a = np.log(actuals + 1)
    log_err = log_p - log_a
    return np.sqrt(log_err.dot(log_err) / len(preds))


def eval_error(preds, dtrain):

    actuals = dtrain.get_label()
    return rmsle_loss(preds, actuals)


if __name__ == '__main__':

    run_based_on_week(train_number=1)




