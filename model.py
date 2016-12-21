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
from const import DTYPES, FEATURES

# Configure logging
logger = logging.getLogger('model')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


use_cols = ['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Demanda_uni_equil']
RAW_TRAIN = pp.load_data(path='raw_data/train.csv', use_cols=use_cols)
RAW_TEST = pp.load_data(path='raw_data/test.csv', use_cols=['id', 'Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID'])


class ClusterXgbModel(object):

    def __init__(self, n_clusters=3, batch_size=1.0, cv_size=0.2):
        self._n_clusters = n_clusters
        self._batch_size = batch_size
        self._cv_size = cv_size
        self._cluster_model = None
        self._U = None
        self._xgb_params = None
        self._xgb_boosters = []

    @staticmethod
    def pca(data, ratio=0.9):
        """
        Simple pca.
        Args:
            data: a ndarray object. Array for pca.
            ratio: float, between 0 and 1, default 0.9. Ratio of the sum of selected principal components.

        Returns:

        """
        data = (data - data.mean(axis=0)) / data.std(axis=0)
        cov = np.dot(data.T, data)
        U, S, V = np.linalg.svd(cov)
        k = 1
        while np.sum(S[: k]) / np.sum(S) < ratio:
            k += 1
        data_r = np.dot(data, U[:, :k])
        return data_r, U[:, :k]

    @staticmethod
    def predict(xgb_reg, data, log_demand=True, normalize=False):

        xgb_reg.set_param({'nthread': 3})

        # prepare data.
        feature_data = data[FEATURES].values
        if normalize:
            feature_data = feature_data / feature_data.max(axis=0)
        dtest = xgb.DMatrix(feature_data, label=None, feature_names=FEATURES)

        # free memory.
        del feature_data

        # predict data set.
        logger.info('Predicting...')
        y_test = xgb_reg.predict(dtest)

        if log_demand:
            data['Demanda_uni_equil'] = np.exp(y_test) - 1
        else:
            data['Demanda_uni_equil'] = y_test

        pred = data[['id', 'Demanda_uni_equil']]  # prediction
        # train data for this week, generally this is the train set for next week.
        train = data[['Semana', 'Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID', 'Demanda_uni_equil']]
        logger.info('Done with prediction.')
        return pred, train

    @property
    def xgb_params(self):
        return self._xgb_params

    @xgb_params.setter
    def xgb_params(self, params):
        self._xgb_params = params

    @property
    def U(self):
        return self._U

    @U.setter
    def U(self, array):
        if isinstance(array, np.ndarray):
            self._U = array
        else:
            raise ValueError('Must be a numpy.ndarray object!')

    def get_cluster_model(self, data):
        """
        Get cluster model for given data.
        Args:
            data: a ndarray object, generally normalized.

        Returns:

        """
        model_path = 'models/cluster_model'
        if not os.path.exists(model_path):
            from sklearn.cluster import KMeans
            cls = KMeans(n_clusters=self._n_clusters, n_jobs=5)
            cls.fit(data)
            pickle.dump(cls, open(model_path, 'wb'))
            self._cluster_model = cls
        else:
            self._cluster_model = pickle.load(open(model_path, 'rb'))

    def set_clusters(self, data):
        """
        Set cluster labels for given data.
        Args:
            data: a DataFrame object contains FEATURE.
        Returns:
        """
        feature_data = data[FEATURES].values
        feature_data = (feature_data - feature_data.mean(axis=0)) / feature_data.std(axis=0)
        pca_feature_data = np.dot(feature_data, self._U)
        del feature_data
        clusters = self._cluster_model.predict(pca_feature_data)
        del pca_feature_data
        data['cluster'] = clusters
        return data

    def train(self, data=None, log_demand=True, normalize=False):
        """
        Train
        Args:
            data: a DataFrame object, default None. If None, 'processed/reg_train.csv' will be loaded.
            log_demand: a boolean object, defult True. If True, the target demand will be transformed by log operation, e.g.:
                        y => log(y+1).
            normalize: a bollean object, default False. If True, the train data will be normalized.

        Returns: A xgb booster.

        """
        # load data
        logger.info('Loading train data...')
        if data is None:
            data = pd.read_csv('processed/reg_train.csv', dtype=DTYPES)

        logger.info('Setting features for training...')

        # sampling
        logger.info('Sampling...')
        data = data.take(np.random.permutation(len(data))[: int(self._batch_size * len(data))])

        # prepare training set.
        logger.info("Preparing training set...")
        if log_demand:
            data.loc[:, 'Demanda_uni_equil'] = np.log(data.Demanda_uni_equil.values + 1)

        x_data = data[FEATURES]
        feature_names = x_data.columns.tolist()
        x_data = x_data.values
        if normalize:
            x_data = x_data / x_data.max(axis=0)
        y_data = data['Demanda_uni_equil'].values

        # free memory
        del data

        # make cross-validation set.
        logger.info('Cross-validation...')
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=self._cv_size)
        dtrain = xgb.DMatrix(data=x_train, label=y_train, feature_names=feature_names)
        dtest = xgb.DMatrix(data=x_test, label=y_test, feature_names=feature_names)

        # free memory
        del x_data, x_train, x_test, y_train, y_test

        print('parameters: \n', self._xgb_params)
        watchlist = [(dtrain, 'train'), (dtest, 'eval')]
        logger.info('Training...')
        xgb_reg = xgb.train(params=self._xgb_params, dtrain=dtrain, num_boost_round=40, evals=watchlist, early_stopping_rounds=5)
        logger.info('Done with training.')
        return xgb_reg

    def train_all(self, data):
        """
        Train xgb models for all clusters.
        Args:
            data: a DataFrame object, default None. data_cls is the dataframe for training, and must
                  be set clusters.

        Returns:

        """
        if 'cluster' not in data.columns.tolist():
            raise AttributeError("data has no column 'cluster'.")
        for cluster in range(0, self._n_clusters):
            model_path = 'models/xgb_cluster_{}.model'.format(cluster)
            logger.info('Loading xgb_cluster_{}.model'.format(cluster))
            if os.path.exists(model_path):
                xgb_reg = xgb.load_model(model_path)
                self._xgb_boosters.append(xgb_reg)
            else:
                logger.info('Model does not exist, training model...')
                _train = data[data.cluster == cluster]
                print('There are {} training samples.'.format(len(_train)))
                xgb_reg = self.train(data=_train)
                xgb_reg.save_model(model_path)
                self._xgb_boosters.append(xgb_reg)

    @staticmethod
    def predict(xgb_reg, data, log_demand=True, normalize=False):

        xgb_reg.set_param({'nthread': 3})

        # prepare data.
        feature_data = data[FEATURES].values
        if normalize:
            feature_data = feature_data / feature_data.max(axis=0)
        dtest = xgb.DMatrix(feature_data, label=None, feature_names=FEATURES)

        # free memory.
        del feature_data

        # predict data set.
        logger.info('Predicting...')
        y_test = xgb_reg.predict(dtest)

        if log_demand:
            data['Demanda_uni_equil'] = np.exp(y_test) - 1
        else:
            data['Demanda_uni_equil'] = y_test

        pred = data[['id', 'Demanda_uni_equil']]  # prediction
        # train data for this week, generally this is the train set for next week.
        train = data[['Semana', 'Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID', 'Demanda_uni_equil']]
        logger.info('Done with prediction.')
        return pred, train

    def run(self):
        """
        Run the whole training and predicting process.
        Returns:

        """
        logger.info('Predicting week 10...')
        logger.info('PCA with week 10...')
        reg_train_10 = pd.read_csv('processed/reg_train_10.csv', dtype=DTYPES)
        reg_train_10 = self.set_clusters(reg_train_10)

        for cluster_no in range(0, self._n_clusters):
            tmp = reg_train_10[reg_train_10.cluster == cluster_no]
            tmp_feature = tmp[FEATURES].values
            dtest = xgb.DMatrix(tmp_feature, label=None, feature_names=FEATURES)
            del tmp_feature
            y_test = self._xgb_boosters[cluster_no].predict(dtest)
            tmp['Demanda_uni_equil'] = np.exp(y_test) - 1

            if cluster_no == 0:
                sub_10 = tmp[['id', 'Demanda_uni_equil']]
                train_10 = tmp[['Semana', 'Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID', 'Demanda_uni_equil']]
            else:
                sub_10 = pd.concat([sub_10, tmp[['id', 'Demanda_uni_equil']]])
                train_10 = pd.concat([train_10, tmp[['Semana', 'Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID',
                                                     'Producto_ID', 'Demanda_uni_equil']]])

        logger.info('Predicting week 11...')
        raw_train = pd.concat([RAW_TRAIN, train_10], axis=0)
        reg_train_11 = pp.extract_lag_features(data_frame=raw_train, test_set=RAW_TEST, week=11)

        logger.info('PCA with week 11...')
        reg_train_11 = self.set_clusters(reg_train_11)

        for cluster_no in range(0, self._n_clusters):
            tmp = reg_train_11[reg_train_11.label == cluster_no]
            tmp_feature = reg_train_11[FEATURES].values
            dtest = xgb.DMatrix(tmp_feature, label=None, feature_names=FEATURES)
            del tmp_feature
            y_test = self._xgb_boosters[cluster_no].predict(dtest)
            tmp['Demanda_uni_equil'] = np.exp(y_test) - 1

            if cluster_no == 0:
                sub_11 = tmp[['id', 'Demanda_uni_equil']]
                train_11 = tmp[['Semana', 'Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID', 'Demanda_uni_equil']]
            else:
                sub_11 = pd.concat([sub_11, tmp[['id', 'Demanda_uni_equil']]])
                train_11 = pd.concat([train_11, tmp[['Semana', 'Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID',
                     'Demanda_uni_equil']]])

        sub = pd.concat([sub_10, sub_11])
        sub = sub.sort_values(by='id')
        sub.to_csv('submission/submission_cluster_xgb.csv', index=False)
        logger.info('Done with submission.')


if __name__ == '__main__':

    # extract weight and piece information of products.
    pp.extract_weight_pieces()

    # extract town id from town_state.csv.
    pp.extract_town_id()

    # extract client count for each town_id.
    pp.extract_client_count_by_town_id(data_frame=RAW_TRAIN)

    # extract price information.
    pp.extract_price(data_frame=RAW_TRAIN)

    # set training set for week 7, 8, 9 and combine them together.
    pp.set_train_data(data_frame=RAW_TRAIN)

    # load regression train data.
    reg_train = pd.read_csv('processed/reg_train.csv', dtype=DTYPES)

    # create cluster-xgb combined model.
    model = ClusterXgbModel()

    # do pca.
    data_r, model.U = model.pca(reg_train[FEATURES].values)
    model.get_cluster_model(data=data_r[:3000000])
    del data_r

    reg_train = model.set_clusters(reg_train)

    # train clustered regression data and get xgb boosters.
    model.train_all(reg_train)

    # run prediction of week 10 and 11.
    model.run()

    # combine cluster_xgb submission with ftrl submission.
    cls_sub = pd.read_csv('submission/submission_cluster_xgb.csv')
    ftrl_sub = pd.read_csv('submission/submission_ftrl.csv')
    final_sub = cls_sub.copy()
    final_sub.loc[:, 'Demanda_uni_equil'] = (cls_sub.Demanda_uni_equil.values + ftrl_sub.Demanda_uni_equil.values) / 2
    final_sub.to_csv('submission/submission_final.csv', index=False, float_format='%.4f')





    

    






