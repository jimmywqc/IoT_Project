# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 14:16:42 2020

@author: Administrator
"""

import pandas as pd
import numpy as np
import zipfile
from sklearn.model_selection import KFold
import lightgbm as lgb

size_train=pd.read_csv('../Train/Size.csv')
df_train_spc=pd.read_csv('../Train/data_spc.csv')
df_test_spc=pd.read_csv('../Test/data_spc.csv')

df_train_spc.drop(columns='remark', inplace=True)
df_test_spc.drop(columns='remark', inplace=True)

variables1=['Sensor1', 'Sensor2', 'Sensor3', 'IJ', 'Sensor5',
       'Sensor6', 'MouldTemp1', 'MouldTemp2', 'MouldTemp3', 'MouldTemp4',
       'MouldTemp5', 'MouldTemp9', 'MouldTemp10', 'MouldTemp11', 'MouldTemp12',
       'MouldTemp13', 'MouldTemp14', 'Sensor8', 'MouldFlow1', 'MouldFlow2',
       'MouldFlow3', 'SP']

################################################################
# 训练集高频数据特征提取
TRAIN_ZIP=zipfile.ZipFile('../Train/传感器高频数据.zip')
file_list=TRAIN_ZIP.namelist()
feature_n = len(variables1)
features_ = np.empty([len(file_list), feature_n])
times_ = []
mold_id_ = []

def feature_columns(variables):
    f_cols = []
    for v in variables:
        f_cols.append(v + '_mean')
    return f_cols

def stage_features(df, variables):
    avg = []
    tmp_df = df.loc[:, variables]
    for v in variables:
        tmp_avg = tmp_df[v].mean()
        avg.append(tmp_avg)    
    return np.array(avg)

feature_n = len(variables1)
features_ = np.empty([len(file_list), feature_n])
times_ = []
mold_id_ = []
for i,f in enumerate(file_list):
    df=pd.read_csv(TRAIN_ZIP.open(f))
    tmp = f.split('_')
    ti = tmp[2]
    mold_id = tmp[3].replace('.csv', '')
    times_.append(str(ti))
    mold_id_.append(int(mold_id))
    if len(df) == 0:
        features_[i] = [None for j in range(feature_n)]
    else:
        features_[i] = stage_features(df, variables1)
f_cols = feature_columns(variables1)
TRAIN_HIG = pd.DataFrame(features_, columns=f_cols)
TRAIN_HIG['Time'] = times_
TRAIN_HIG['Id'] = mold_id_
TRAIN_HIG = TRAIN_HIG[['Id', 'Time'] + f_cols]
TRAIN_ZIP.close()

################################################################
# 测试集高频数据特征提取
TEST_ZIP=zipfile.ZipFile('../Test/传感器高频数据.zip')
file_list=TEST_ZIP.namelist()
feature_n = len(variables1)
features_ = np.empty([len(file_list), feature_n])
times_ = []
mold_id_ = []
for i,f in enumerate(file_list):
    df=pd.read_csv(TEST_ZIP.open(f))
    tmp = f.split('_')
    ti = tmp[2]
    mold_id = tmp[3].replace('.csv', '')
    times_.append(str(ti))
    mold_id_.append(int(mold_id))
    if len(df) == 0:
        features_[i] = [None for j in range(feature_n)]
    else:
        features_[i] = stage_features(df, variables1)
f_cols = feature_columns(variables1)
TEST_HIG = pd.DataFrame(features_, columns=f_cols)
TEST_HIG['Time'] = times_
TEST_HIG['Id'] = mold_id_
TEST_HIG = TEST_HIG[['Id', 'Time'] + f_cols]
TEST_ZIP.close()

TRAIN_HIG.rename(columns={'Time':'spcTime'}, inplace=True)
TEST_HIG.rename(columns={'Time':'spcTime'}, inplace=True)

df_train_spc['spcTime']=df_train_spc['spcTime'].apply(int)
df_train_spc['spcTime']=df_train_spc['spcTime'].apply(str)
df_test_spc['spcTime']=df_test_spc['spcTime'].apply(int)
df_test_spc['spcTime']=df_test_spc['spcTime'].apply(str)

df_TRAIN=TRAIN_HIG.merge(df_train_spc, how='outer', on=['Id', 'spcTime']) 
df_TEST=TEST_HIG.merge(df_test_spc, how='outer', on=['Id', 'spcTime']) 
X_col=[i for i in df_TRAIN.columns if not i in ['Id', 'spcTime']]

################################################################
# 模型
K = 5
seed = 1234
skf = KFold(n_splits=K, shuffle=True, random_state=seed)
lgb_params = {
                        'boosting_type': 'gbdt',
                        'objective': 'regression',
                        'num_leaves': 2**5,
                        'subsample': 0.9,
                        'learning_rate': 0.05,
                        'seed': 2017,
                        'nthread': -1
             }

def mode_(size_i):
    predictions = np.zeros(len(X_test))
    for i, (train_index, val_index) in enumerate(skf.split(X_train,y_train)):
        print("fold {}".format(i))
        X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]
        lgb_train = lgb.Dataset(X_tr,y_tr)
        lgb_val = lgb.Dataset(X_val,y_val)
        num_round = 2000
        clf = lgb.train(lgb_params, lgb_train, num_round, valid_sets = [lgb_train, lgb_val],verbose_eval=50, 
                        early_stopping_rounds = 50)
        print('best iteration = ',clf.best_iteration)
        print("*"*100)
        predictions += clf.predict(X_test, num_iteration=clf.best_iteration) / skf.n_splits
    return predictions

sub=pd.read_csv('../Test初赛/sub_file.csv')
for i in ['size1','size2','size3']:
    print(i)
    X_train=df_TRAIN[X_col]
    y_train=size_train[i]
    X_test=df_TEST[X_col]
    sub[i]=mode_(i)

# sub.to_csv('./pred.csv', index=False)