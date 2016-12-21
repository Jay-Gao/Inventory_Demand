#! ~/anaconda3/bin/python3
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a python script for pre-process of data.
"""


import os
import re
import pickle
import logging
import numpy as np
import pandas as pd


# Configure logging
logger = logging.getLogger('preprocess')
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


def load_data(path='raw_data/train.csv', use_cols=None):
    """
    load data from csv file with given columns.
    Args:
        path: csv file path.
        use_cols: column that you want to load.

    Returns: DataFrame object.
    """

    dtypes = {'Semana': 'int32',
             'Agencia_ID': 'int32',
             'Canal_ID': 'int32',
             'Ruta_SAK': 'int32',
             'Producto_ID': 'int32',
             'Cliente_ID': 'int32',
             'NombreCliente': 'object',
             'NombreProducto': 'object',
             'Venta_uni_hoy': 'int32',
             'Venta_hoy': 'float32',
             'Dev_uni_proxima': 'int32',
             'Dev_proxima': 'float32',
             'Demanda_uni_equil': 'float32'
              }

    if use_cols is None:
        use_cols = []
    cols = ['Semana', 'Producto_ID']
    cols.extend(use_cols)
    return pd.read_csv(filepath_or_buffer=path, dtype=dtypes, usecols=cols)


def extract_weight_pieces():
    """
    Extract basic product information from producto_table.csv, including:
    weight, pieces, weight per piece and brand of the products.
    Returns:

    """
    # extract weight information based on 'g', 'kg' or 'Kg'.
    def weight(x):
        match = re.findall('\s(\d+)([Kk]*g)\s', x)
        if match:
            # if unit is kg or Kg, then multiply weight by 1000.
            if match[0][1] in ['Kg', 'kg']:
                return 1000 * int(match[0][0])
            else:
                return int(match[0][0])
        else:
            return None

    # extract pieces information based on 'p'.
    def pieces(x):
        match = re.findall('\s(\d+)p\s', x)
        if match:
            return int(match[0])
        else:
            return None

    # load products' description file.
    product_info = pd.read_csv('raw_data/producto_tabla.csv')
    # replace 'ml' with 'g'.
    product_info.loc[:, 'NombreProducto'] = product_info['NombreProducto'].apply(lambda x: x.replace('ml', 'g'))
    product_info['weight'] = product_info['NombreProducto'].apply(lambda x: weight(x))
    # fill na with value.
    product_info.ix[:, 'weight'] = product_info['weight'].fillna(value=product_info.weight.mean())

    product_info['pieces'] = product_info['NombreProducto'].apply(lambda x: pieces(x))
    product_info.ix[:, 'pieces'] = product_info['pieces'].fillna(value=1.0)
    product_info['weight_per_piece'] = product_info.weight / product_info.pieces
    product_info['brand'] = product_info['NombreProducto'].apply(lambda x: x.split()[-2])
    product_info.to_csv('new_data/products.csv', index=False, float_format='%.0f')


def extract_town_id():
    """
    Extract town id from town_state.csv.
    Returns:

    """
    town_info = pd.read_csv('raw_data/town_state.csv')
    town_info['town_id'] = town_info.Town.apply(lambda x: int(x.split()[0]))
    town_info.to_csv('new_data/town_info.csv', index=False)


# %%set features
# def _min(col): return col.values[: -1].min()
# def _max(col): return col.values[: -1].max()
# def _mean(col): return col.values[: -1].mean()
# def _std(col): return col.values[: -1].std()
# def _last_week(col): return col.values[-2]
# def _trend(col): return col.values[-1] / col.values[0] - 1
# def _price(col): return col.values[: -1].mean()
# def _target(col): return col.values[-1]  # the target for training set.
# _min.__name__ = 'min'
# _max.__name__ = 'max'
# _mean.__name__ = 'mean'
# _std.__name__ = 'std'
# _last_week.__name__ = 'last_week'
# _trend.__name__ = 'trend'
# _price.__name__ = 'price'
# _target.__name__ = 'target'


# %% features functions
# def demand_based_on_product_id(data_frame=None):
#
#     if data_frame is None:
#         use_cols = ['Demanda_uni_equil']
#         data_frame = load_data(path='raw_data/train.csv', use_cols=use_cols)
#
#     mean_demand = data_frame.groupby('Producto_ID')['Demanda_uni_equil'].mean()
#     mean_demand.to_frame(name='mean_demand').reset_index().to_csv('data/productid_demand.csv', index=False, float_format='%.4f')
#     pickle.dump(mean_demand.to_dict(), open('data/productid_demand.dict', 'wb'))
#
#
# def sales_based_on_product_id(data_frame=None):
#
#     if data_frame is None:
#         use_cols = ['Venta_uni_hoy']
#         data_frame = load_data(path='raw_data/train.csv', use_cols=use_cols)
#
#     mean_sales = data_frame.groupby('Producto_ID')['Venta_uni_hoy'].mean()
#     mean_sales.to_frame(name='mean_sales').reset_index().to_csv('data/productid_sales.csv', index=False, float_format='%.4f')
#     pickle.dump(mean_sales.to_dict(), open('data/productid_sales.dict', 'wb'))
#
#
# def depot_based_on_product_id(data_frame=None):
#
#     if data_frame is None:
#         use_cols = ['Agencia_ID']
#         data_frame = load_data(path='raw_data/train.csv', use_cols=use_cols)
#
#     depot_num = data_frame.groupby('Producto_ID')['Agencia_ID'].unique().apply(len)
#     depot_num.to_frame(name='depot_num').reset_index().to_csv('data/productid_depot.csv', index=False, float_format='%.4f')
#     pickle.dump(depot_num.to_dict(), open('data/productid_depot.dict', 'wb'))
#
#
# def channel_based_on_product_id(data_frame=None):
#
#     if data_frame is None:
#         use_cols = ['Canal_ID']
#         data_frame = load_data(path='raw_data/train.csv', use_cols=use_cols)
#
#     channel_num = data_frame.groupby('Producto_ID')['Canal_ID'].unique().apply(len)
#     channel_num.to_frame(name='channel_num').reset_index().to_csv('data/productid_channel.csv',
#                                                                   index=False, float_format='%.4f')
#     pickle.dump(channel_num.to_dict(), open('data/productid_channel.dict', 'wb'))
#
#
# def route_based_on_product_id(data_frame=None):
#
#     if data_frame is None:
#         use_cols = ['Ruta_SAK']
#         data_frame = load_data(path='raw_data/train.csv', use_cols=use_cols)
#
#     route_num = data_frame.groupby('Producto_ID')['Ruta_SAK'].unique().apply(len)
#     route_num.to_frame(name='route_num').reset_index().to_csv('data/productid_route.csv',
#                                                                   index=False, float_format='%.4f')
#     pickle.dump(route_num.to_dict(), open('data/productid_route.dict', 'wb'))
#
#
# def client_based_on_product_id(data_frame=None):
#
#     if data_frame is None:
#         use_cols = ['Cliente_ID']
#         data_frame = load_data(path='raw_data/train.csv', use_cols=use_cols)
#
#     client_num = data_frame.groupby('Producto_ID')['Cliente_ID'].unique().apply(len)
#     client_num.to_frame(name='client_num').reset_index().to_csv('data/productid_client.csv',
#                                                               index=False, float_format='%.4f')
#     pickle.dump(client_num.to_dict(), open('data/productid_client.dict', 'wb'))
#
#
# def extract_features(data_frame=None, week=7):
#     """
#     extract features from given data frame at certain week.
#     Args:
#         data_frame: pandas.DataFrame.
#         week: int, week you want extract feature for.
#     Returns:
#     """
#     if data_frame is None:
#         use_cols = ['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Venta_uni_hoy', 'Venta_hoy', 'Demanda_uni_equil']
#         data_frame = load_data(path='raw_data/train.csv', use_cols=use_cols)
#
#     # select data with Semana < week.
#     data_frame = data_frame[data_frame['Semana'] > week-5]
#     data_frame = data_frame[data_frame['Semana'] < week+1]
#     print('Done with selecting data.')
#
#     # set price, with large amount of math, it better to implement it with numpy.ndarray.
#     sales = data_frame['Venta_hoy'].values
#     amount = data_frame['Venta_uni_hoy'].values
#     data_frame['price'] = sales / amount
#     del sales, amount
#
#     # setting features.
#     demand_features = [_min, _max, _mean, _std, _trend, _last_week]
#     # demand_features = [_min]
#     price_features = [_price]
#     print('Done with setting features.')
#
#     # extract features.
#     # there must be some other way to optimize this.
#     group_cols = ['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID']
#     count = data_frame.groupby(group_cols)['Demanda_uni_equil'].agg(['count']).reset_index()
#     data_frame = data_frame.merge(count, on=group_cols)
#
#     # release memory
#     del count
#
#     data_frame = data_frame[data_frame['count'] > 2].drop('count', axis=1)
#     print('Done with select group.')
#
#     df1 = data_frame.groupby(group_cols)['Demanda_uni_equil'].agg(demand_features)
#     print('Done with extracting features.')
#     df2 = data_frame.groupby(group_cols)['price'].agg(price_features)
#     # fill None price with mean value.
#     mean_value = df2.mean()
#     df2 = df2.fillna(value=mean_value)
#     data_frame = df1.merge(df2, left_index=True, right_index=True).reset_index()
#     print('Done with setting price.')
#
#     # add features with merge based on Product ID.
#     mean_demand = pd.read_csv('data/productid_demand.csv')
#     mean_sales = pd.read_csv('data/productid_sales.csv')
#     depot_num = pd.read_csv('data/productid_depot.csv')
#     channel_num = pd.read_csv('data/productid_channel.csv')
#     route_num = pd.read_csv('data/productid_route.csv')
#     client_num = pd.read_csv('data/productid_client.csv')
#
#     data_frame = data_frame.merge(mean_demand, on='Producto_ID')
#     data_frame = data_frame.merge(mean_sales, on='Producto_ID')
#     data_frame = data_frame.merge(depot_num, on='Producto_ID')
#     data_frame = data_frame.merge(channel_num, on='Producto_ID')
#     data_frame = data_frame.merge(route_num, on='Producto_ID')
#     data_frame = data_frame.merge(client_num, on='Producto_ID')
#     print('Done with merging Product ID.')
#
#     return_cols = ['min', 'max', 'mean', 'std', 'trend', 'last_week', 'price', 'mean_demand', 'mean_sales',
#                    'depot_num', 'channel_num', 'route_num', 'client_num', 'target']
#     return data_frame[return_cols]
#
#
# def set_reg_train(data_frame=None, weeks=None):
#
#     if weeks is None:
#         weeks = [7, 8, 9]
#     if not os.path.exists('data/reg_train.csv'):
#         week = weeks[0]
#         out_df = extract_features(data_frame=data_frame, week=week)
#         out_df.to_csv('data/reg_train.csv', mode='w', index=False, header=True, float_format='%.4f')
#         print('Done with week {}'.format(week))
#         for week in weeks[1:]:
#             out_df = extract_features(data_frame=data_frame, week=week)
#             out_df.to_csv('data/reg_train.csv', mode='a', index=False, header=False, float_format='%.4f')
#             print('Done with week {}'.format(week))
#
#
# def set_reg_test(data_frame=None, week=9):
#
#     if not os.path.exists('data/reg_test.csv'):
#         out_df = extract_features(data_frame=data_frame, week=week)
#         out_df.to_csv('data/reg_test.csv', mode='w', index=False, header=True, float_format='%.4f')
#     print('Done with week {}'.format(week))


#----------------------------------------------------------------------------------------------------------------------#
# new way to extract features.
# extract lag features for product-client pairs

def extract_features(data_frame=None, test_set=None,  week=7, log_demand=False, lag_start=1):
    """
    extract features including lag demand, price information etc.
    Args:
        data_frame: DataFrame object, default None. If None, train.csv will be loaded.
        test_set: DataFrame object, default None. If not None, it will be used to extract product-client pairs and
                  generally this is for week 10 and 11.
        week: int, dufault 7. The week you want extract features for. If test_set is not None, week must be 10 or 11,
              else it should be 7, 8 or 9.
        log_demand: boolean, default False. If True, the lag feature will be average of log-demands of certain week, e.g.:
                    mean(log(demand + 1)), else the lag feature will be median of demands of certain week.
        lag_start: int, default 1. The lag feature you want to start from.
    Returns: DataFrame object. The feature dataframe extracted for given week.

    """
    # load data.
    if data_frame is None:
        use_cols = ['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Demanda_uni_equil']
        data_frame = load_data(path='raw_data/train.csv', use_cols=use_cols)
    logger.info('Done with loading data frame.')

    # set product-client pairs.
    if test_set is None:
        week_df = data_frame[data_frame.Semana == week]
    else:
        week_df = test_set[test_set.Semana == week]
    product_client_pairs = week_df.groupby(['Producto_ID', 'Cliente_ID'])['Agencia_ID'].agg\
                                                                        ([('dummy', lambda col: 0.0)]).reset_index()
    logger.info('Done with setting product-client pairs.')

    if log_demand:
        logger.info('log is True, transforming demand data.')
        data_frame['Demanda_uni_equil'] = np.log(data_frame['Demanda_uni_equil'].values + 1)

    # extract lag features.
    for lag in range(lag_start, 5):
        df = data_frame[data_frame.Semana == week - lag]
        lag_name = 'lag{}'.format(lag)
        if log_demand:
            df = df.groupby(['Producto_ID', 'Cliente_ID'])['Demanda_uni_equil'].agg([(lag_name, 'mean')]).reset_index()
        else:
            df = df.groupby(['Producto_ID', 'Cliente_ID'])['Demanda_uni_equil'].agg([(lag_name, 'median')]).reset_index()
        product_client_pairs = product_client_pairs.merge(df, how='left', on=['Producto_ID', 'Cliente_ID'])
        product_client_pairs.loc[:, lag_name] = product_client_pairs[lag_name].fillna(value=0.0)
        logger.info('Done with lag{}'.format(lag))
    # remove dummy column
    product_client_pairs = product_client_pairs.drop('dummy', axis=1)

    # add sum of lag features.
    lag_array = product_client_pairs[['lag{}'.format(i) for i in range(lag_start, 5)]].values
    product_client_pairs['sum_of_lag_features'] = lag_array.sum(axis=1)
    del lag_array
    logger.info('Done with adding sum of lags.')

    # add sum and mean of prior demand based on product-client pairs.
    week_before = data_frame[data_frame.Semana < week-lag_start+1]
    prior_demand = week_before.groupby(['Producto_ID', 'Cliente_ID'])['Demanda_uni_equil'].agg\
                                    ([('sum_of_prior_demand', 'sum'), ('mean_of_prior_demand', 'mean')]).reset_index()
    product_client_pairs = product_client_pairs.merge(prior_demand, how='left', on=['Producto_ID', 'Cliente_ID'])
    product_client_pairs.ix[:, 'sum_of_prior_demand'] = product_client_pairs['sum_of_prior_demand'].fillna(value=0.0)
    product_client_pairs.ix[:, 'mean_of_prior_demand'] = product_client_pairs['mean_of_prior_demand'].fillna(value=0.0)
    del week_before, prior_demand
    logger.info('Done with adding sum and mean of prior demand.')

    # add frequent of product-client pairs in this week.
    freq = week_df.groupby(['Producto_ID', 'Cliente_ID'])['Agencia_ID'].agg\
                                                    ([('product_client_pairs_freq', 'count')]).reset_index()
    product_client_pairs = product_client_pairs.merge(freq, how='left', on=['Producto_ID', 'Cliente_ID'])
    product_client_pairs.ix[:, 'product_client_pairs_freq'] = product_client_pairs['product_client_pairs_freq'].fillna(value=0)
    logger.info('Done with adding frequent of product-client pairs.')

    # add client count by town id.
    if not os.path.exists('new_data/client_count_by_town_id.csv'):
        extract_client_count_by_town_id(data_frame=data_frame)
    client_count_by_town_id = pd.read_csv('new_data/client_count_by_town_id.csv', usecols=['Agencia_ID', 'client_count'],\
                                          dtype={'Agencia_ID': 'int32', 'client_count': 'int32'})
    week_df = week_df.merge(client_count_by_town_id, on='Agencia_ID')
    logger.info('Done with adding client count.')

    # merge product-client pairs features to this week.
    week_df = week_df.merge(product_client_pairs, on=['Producto_ID', 'Cliente_ID'])
    logger.info('Done with merging product-client pairs.')

    # add products' information of weight, pieces, weight per piece and brand.
    products_info = pd.read_csv('new_data/product_info.csv', usecols=['Producto_ID', 'price', 'weight', 'pieces',
                                                                               'weight_per_piece', 'brand'])
    week_df = week_df.merge(products_info, how='left', on='Producto_ID')
    logger.info('Done with adding products information.')

    return week_df


def set_train_data(data_frame=None, weeks=None, train_number=1, log_demand=False, lag_start=1):

    if weeks is None:
        weeks = [7, 8, 9]

    path = 'new_data/reg_train_{:02}.csv'.format(train_number)

    if not os.path.exists(path):
        week = weeks[0]
        week_df = extract_features(data_frame=data_frame, week=week, log_demand=log_demand, lag_start=lag_start)
        week_df.to_csv(path, mode='w', index=False, header=True, float_format='%.4f')
        logger.info('Done with week {}'.format(week))
        for week in weeks[1:]:
            week_df = extract_features(data_frame=data_frame, week=week, log_demand=log_demand, lag_start=lag_start)
            week_df.to_csv(path, mode='a', index=False, header=False, float_format='%.4f')
            logger.info('Done with week {}'.format(week))


def set_test_10(train_number, log_demand=False):

    use_cols = ['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Demanda_uni_equil']
    train = load_data(path='raw_data/train.csv', use_cols=use_cols)
    test_set = pd.read_csv('raw_data/test.csv')

    week_10 = extract_lag_features(data_frame=train, test_set=test_set, week=10, log_demand=log_demand)

    week_10.to_csv('new_data/test_week_10_{:02}.csv'.format(train_number), index=False, header=True, float_format='%.4f')


def set_test_11(train_number, log_demand=False, lag_start=1):

    use_cols = ['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Demanda_uni_equil']
    # week_10 = load_data(path='new_data/train_10.csv', use_cols=use_cols)

    train = load_data(path='raw_data/train.csv', use_cols=use_cols)
    # train = pd.concat([train, week_10], axis=0)
    test_set = pd.read_csv('raw_data/test.csv')
    # del week_10
    week_11 = extract_lag_features(data_frame=train, test_set=test_set, week=11, log_demand=log_demand, lag_start=lag_start)
    week_11.to_csv('new_data/test_week_11_{:02}.csv'.format(train_number), index=False, header=True, float_format='%.4f')


def extract_client_count_by_town_id(data_frame=None):

    if data_frame is None:
        use_cols = ['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Venta_uni_hoy', 'Venta_hoy', 'Demanda_uni_equil']
        data_frame = load_data(path='raw_data/train.csv', use_cols=use_cols)
    print('Done with loading data frame.')

    town_info = pd.read_csv('new_data/town_info.csv', usecols=['Agencia_ID', 'town_id'], dtype={'Agencia_ID': 'int32', 'town_id': 'int32'})
    data_frame = data_frame.merge(town_info, on='Agencia_ID')
    client_count_by_town_id = data_frame.groupby('town_id')['Cliente_ID'].agg([('client_count', 'count')]).reset_index()
    client_count_by_town_id = client_count_by_town_id.merge(town_info, on='town_id')
    client_count_by_town_id.to_csv('new_data/client_count_by_town_id.csv', index=False)


def pca_analysis(data):

    data = (data - data.mean(axis=0)) / data.std(axis=0)
    cov = np.dot(data.T, data)
    U, S, V = np.linalg.svd(cov)
    data_r = np.dot(data, U[:, :5])
    return data_r, U[:, :5]


def kmeans(data):

    if not os.path.exists('models/kmeans_model'):
        from sklearn.cluster import KMeans
        cls = KMeans(n_clusters=3, n_jobs=5)
        cls.fit(data)
        pickle.dump(cls, open('models/kmeans_model', 'wb'))


def cluster_predict():

    train = pd.read_csv('new_data/reg_train_01.csv', dtype=DTYPES)
    feature_data = train[FEATURES].values

    pca_feature_data = pca_analysis(feature_data)
    cls = pickle.load(open('models/kmeans_model', 'rb'))

    labels = cls.predict(pca_feature_data)

    train['label'] = labels
    train.to_csv('new_data/reg_train_01_cls.csv', index=False)



if __name__ == '__main__':

    pass