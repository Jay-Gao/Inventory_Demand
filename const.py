#! ~/anaconda3/bin/python3
# -*- coding: utf-8 -*-

# data tyoe dict.
DTYPES = {'Agencia_ID': 'int32', 'Canal_ID': 'int32', 'Ruta_SAK': 'int32', 'Cliente_ID': 'int32', 'Producto_ID': 'int32',
          'Demanda_uni_equil': 'float32', 'client_count': 'int32', 'lag1': 'float32', 'lag2': 'float32', 'lag3': 'float32',
          'lag4': 'float32', 'sum_of_lag_features': 'float32', 'sum_of_prior_demand': 'float32','mean_of_prior_demand': 'float32',
          'product_client_pairs_freq': 'int32', 'price': 'float32', 'weight': 'float32', 'pieces': 'int32',
          'weight_per_piece': 'float32', 'brand': 'object'}

# features for training and prediction.
FEATURES = ['client_count', 'lag1', 'lag2', 'lag3', 'lag4', 'sum_of_lag_features', 'sum_of_prior_demand',
             'mean_of_prior_demand', 'product_client_pairs_freq', 'price', 'weight', 'pieces', 'weight_per_piece']