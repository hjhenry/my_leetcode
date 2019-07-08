# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import h5py
data_dir = os.getcwd()
from factor_test_tools import get_ic_ts
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    #import statsmodels.api as sm
from sklearn import metrics
from sklearn import svm
import cvxpy as cvx
"""
找到一种方法能够稳定战胜过去12个月的IC_mean,IR加权的方法
模型：
1.非线性
2.线性，稀疏
3.线性回归

模型1：单因子对单因子
模型2：多因子对单因子
"""


"""
月度以及周度的IC_ts
"""
 
#alpha_factor_sel_dict.keys()
alpha_factor_sel_dict = {}
domain = os.path.join(data_dir, 'alpha_factor_db', 'alpha_neu_std_1')   
name_list = []
for i, name in enumerate(os.listdir(domain)):
    name_list.append(name[:-4])
    #    if i in a_list:
    if i in  a_list:
        print(i, name[:-4])
        info = os.path.join(domain, name)
        data_std = pd.read_csv(info, index_col=0)
        data_std = data_std.loc["2007-05-30":]
        alpha_factor_sel_dict[name[:-4]] = data_std.shift(1)
        
alpha_factor_sel_dict.keys()

stock_return_forward_panel = pd.read_hdf(os.path.join(data_dir, 'basic_data.h5'), 'stock_return_forward_panel')
IC_ts_month = get_ic_ts(alpha_factor_sel_dict, stock_return_forward_panel.loc[:, "2007-05-30":, :], period='month', rank=True)
#IC_ts_week = get_ic_ts(alpha_factor_sel_dict, stock_return_forward_panel.loc[:, "2007-05-30":, :], period='week', rank=True)
IC_ts_month.to_csv(os.path.join(data_dir, "weighting", "ml_predIC", "IC_ts_month.csv"))
#IC_ts_week.to_csv(os.path.join(data_dir, "weighting", "ml_predIC", "IC_ts_week.csv"))





"""
做成特征, 形成X_train, y_train, X_test, y_test
month级的数据
"""

# 单因子IC相关feature
IC_ts_month = pd.read_csv(os.path.join(data_dir, "weighting", "ml_predIC", "IC_ts_month.csv"), index_col=0)
IC_ts_week = pd.read_csv(os.path.join(data_dir, "weighting", "ml_predIC", "IC_ts_week.csv"), index_col=0)
IC_ts_month.index = pd.DatetimeIndex(IC_ts_month.index)
index_return = pd.read_hdf(os.path.join(data_dir, 'basic_data.h5'), 'index_return')
index_return = index_return.loc[IC_ts_month.index, 'wind_A']

IC_ts_month = IC_ts_month.iloc[0:-1:20]
IC_ts_month.index = pd.DatetimeIndex(IC_ts_month.index)

IC_mean_1m = IC_ts_month.rolling(window=1, min_periods=1).mean().shift(1)
IC_mean_3m = IC_ts_month.rolling(window=3, min_periods=3).mean().shift(1)
IC_mean_6m = IC_ts_month.rolling(window=6, min_periods=6).mean().shift(1)
IC_mean_12m = IC_ts_month.rolling(window=12, min_periods=12).mean().shift(1)
IC_std_6m = IC_ts_month.rolling(window=6, min_periods=6).std().shift(1)
IC_std_12m = IC_ts_month.rolling(window=12, min_periods=12).std().shift(1)

#IC_delta_1_3 = IC_mean_1m - IC_mean_3m
#IC_delta_3_6 = IC_mean_3m - IC_mean_6m
#IC_delta_6_12 = IC_mean_6m - IC_mean_12m

len(IC_ts_month)
# 市场涨跌，换手率，波动性
index_return_20d = (np.exp(np.log(index_return+1).rolling(window=20, min_periods=20).sum())-1)
index_return_20d = index_return_20d.shift(1).iloc[0:-1:20]
#index_return_20d = index_return_20d.loc[IC_ts_month]

index_std_20d = index_return.rolling(window=20, min_periods=20).std()
index_std_20d = index_std_20d.shift(1).iloc[0:-1:20]


#mkt_status = index_return_20d>0.05
#mkt_status = sum(index_return_20d<-0.05)/float(len(index_return_20d))

#index_return_20d.index = pd.DatetimeIndex(index_return_20d.index)
#index_return_20d.index
#len(index_return_20d.loc[IC_ts_month.index])

# examples
#index_return_20d.corr(IC_ts_month.loc[:, "ROE_q_orth"], method="spearman")
#IC_mean_6m.loc[:,"ROE_q_orth"].corr(IC_mean_3m.loc[:, "ROE_q_orth"], method="spearman")
#IC_std_12m.loc[:,"ROE_q_orth"].corr(IC_ts_month.loc[:, "ROE_q_orth"], method="spearman")
#IC_delta_6_12.loc[:,"ROE_q_orth"].corr(IC_ts_month.loc[:, "ROE_q_orth"], method="spearman")
#
#
#index_std_20d.corr(IC_ts_month.loc[:, 'TURNOVER1M_orth'], method="spearman")
#IC_delta_1_3.loc[:,'TURNOVER1M_orth'].corr(IC_mean_1m.loc[:, 'TURNOVER1M_orth'], method="spearman")
#IC_delta_3_6.loc[:,'TURNOVER1M_orth'].corr(IC_ts_month.loc[:, 'TURNOVER1M_orth'], method="spearman")
#IC_mean_12m.loc[:,'TURNOVER1M_orth'].corr(IC_ts_month.loc[:, 'TURNOVER1M_orth'], method="spearman")

"""
需要三个结果
"""
y_hat_df = pd.DataFrame(np.nan, index=IC_ts_month.index, columns=IC_ts_month.columns)
y_hat_df_lr = pd.DataFrame(np.nan, index=IC_ts_month.index, columns=IC_ts_month.columns)
methods_metrics = ["y_hat_corr", "y_hat_corr_lr", "base_corr", "y_hat_rank", "y_hat_rank_lr",  "base_corr_rank", 
                   "y_hat_ase", "y_hat_ase_lr", "base_ase"]
metric_df = pd.DataFrame(np.nan, index=methods_metrics, columns=IC_ts_month.columns)

#y_hat_df = pd.DataFrame(np.nan, index=[], columns=IC_ts_month.columns)







"""
制作X的例子:ROE
"""
for factor_name in IC_ts_month.columns:
#factor_name = 'SWING1M_orth'
    X = pd.DataFrame(np.nan, index = IC_mean_1m.index, columns=[])
    X.loc[:, "IC_mean_1m"] = IC_mean_1m.loc[:, factor_name]
    X.loc[:, "IC_mean_3m"] = IC_mean_3m.loc[:, factor_name]
    X.loc[:, "IC_mean_6m"] = IC_mean_6m.loc[:, factor_name]
    X.loc[:, "IC_mean_12m"] = IC_mean_12m.loc[:,factor_name]
    X.loc[:, "IC_std_6m"] = IC_std_6m.loc[:,factor_name]
    X.loc[:, "IC_std_12m"] =IC_std_12m.loc[:,factor_name]
    X.loc[:, "index_return_20d"] = index_return_20d
    X.loc[:, "index_std_20d"] = index_std_20d
    
    y = IC_ts_month.loc[:, factor_name]
    
    active_date = X.notnull().all(1) * y.notnull()
    X = X.loc[active_date]
    y = y.loc[active_date]
    
    train_len = len(X) - 36
    y_hat_df.loc[y[train_len:].index, factor_name] = 0
    y_hat_df_lr.loc[y[train_len:].index, factor_name] = 0
    """
    特征处理
    """






    """
    简单的例子
    lr ridge rf gbrt
    1.固定时段 
    2.rolling
    """

    
    #X.columns
    
    #test_t = 37
    # 固定时段
    lr = linear_model.LinearRegression(normalize=True)
    lr.fit(X.iloc[:train_len], y[:train_len])
    y_hat_lr = lr.predict(X.iloc[train_len:])
#    model = sm.OLS(y[:train_len], X.iloc[:train_len])
#    results = model.fit()
#    results.summary()
    for i in range(len(lr.coef_)):
        print(X.columns[i], lr.coef_[i])
#print("intercept", lr.intercept_)


#print("base_corr')
#(X.ix[train_len:, "IC_mean_12m"]).corr(y[train_len:], method="spearman")
#
#y[train_len:].plot.bar()
#(X.ix[train_len:, "IC_mean_12m"]/X.ix[train_len:, "IC_std_12m"]).plot.bar()
#(X.ix[train_len:, "IC_mean_12m"]).plot.bar()
#pd.DataFrame(y_hat).plot.bar()

    """
    gbrt
    """
    gbrt = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=10, random_state=2, loss='lad')
    gbrt.fit(X.iloc[:train_len], y[:train_len])
    y_hat_gbrt = gbrt.predict(X.iloc[train_len:])
    gbrt.feature_importances_

#ridgeCV = RidgeCV(alphas)
#ridgeCV.fit(X.iloc[:train_len],  y[:train_len])
#ridgeCV.alpha_
#
#ridge = Ridge(alpha=0.1, copy_X=True, fit_intercept=False, max_iter=None, normalize=False, 
#              random_state=None, solver='auto', tol=0.001).fit(X.iloc[:train_len], y[:train_len])
#y_hat = ridge.predict(X.iloc[train_len:])
#ridge.coef_

    """
    random forest
    """
    rf = RandomForestRegressor(criterion='mse', max_depth=10, max_features='auto', n_estimators=100, random_state=2, 
                               verbose=0)
    rf.fit(X.iloc[:train_len], y[:train_len])
    y_hat_rf = rf.predict(X.iloc[train_len:])
    rf.feature_importances_
    
    """
    svr
    """
    svr = svm.SVR(C=1.0, kernel='rbf', verbose=True)
    svr.fit(X.iloc[:train_len], y[:train_len])
    y_hat_svr = svr.predict(X.iloc[train_len:])


    """
    评价指标
    corr
    rmse
    abs_mean
    """
    y_hat = (y_hat_lr + y_hat_rf + y_hat_gbrt + y_hat_svr)/4
    #y_hat = y_hat_lr
    y_hat = pd.Series(y_hat, index=y[train_len:].index)
    y_hat_lr = pd.Series(y_hat_lr, index=y[train_len:].index)
    
    y_hat_df.loc[y_hat.index, factor_name] = y_hat.values
    y_hat_df_lr.loc[y_hat_lr.index, factor_name] = y_hat_lr.values
    
    metric_df.loc["y_hat_corr", factor_name] =  y_hat.corr(y[train_len:])
    metric_df.loc["y_hat_corr_lr", factor_name] =  y_hat_lr.corr(y[train_len:])
    metric_df.loc["base_corr", factor_name] =  (X.ix[train_len:, "IC_mean_12m"]/X.ix[train_len:, "IC_std_12m"]).corr(y[train_len:])
    metric_df.loc["base_corr_1", factor_name] =  (X.ix[train_len:, "IC_mean_12m"]).corr(y[train_len:])
    
    metric_df.loc["y_hat_rank", factor_name] =  y_hat.corr(y[train_len:], method="spearman")
    metric_df.loc["y_hat_rank_lr", factor_name] =  y_hat_lr.corr(y[train_len:], method="spearman")
    metric_df.loc["base_corr_rank", factor_name] =  (X.ix[train_len:, "IC_mean_12m"]/X.ix[train_len:, "IC_std_12m"]).corr(y[train_len:], 
                 method="spearman")
    
    metric_df.loc["y_hat_ase", factor_name] =  metrics.mean_absolute_error(y_hat, y[train_len:])
    metric_df.loc["y_hat_ase_lr", factor_name] =  metrics.mean_absolute_error(y_hat_lr, y[train_len:])
    metric_df.loc["base_ase", factor_name] =  metrics.mean_absolute_error(X.ix[train_len:, "IC_mean_12m"],y[train_len:])
    
    print("finish:" + factor_name)


#print("y_hat_corr", y_hat.corr(y[train_len:]))
#print("y_hat_rank", y_hat.corr(y[train_len:], method="spearman"))
#print("base_corr", (X.ix[train_len:, "IC_mean_12m"]/X.ix[train_len:, "IC_std_12m"]).corr(y[train_len:]))
#print("base_corr_rank", (X.ix[train_len:, "IC_mean_12m"]/X.ix[train_len:, "IC_std_12m"]).corr(y[train_len:], 
#      method="spearman"))
#print("y_hat_ase", metrics.mean_absolute_error(y_hat, y[train_len:]))
#print("base_ase", metrics.mean_absolute_error(X.ix[train_len:, "IC_mean_12m"],y[train_len:]))

# plot
    
y[train_len:].plot.bar()
pd.DataFrame(y_hat).plot.bar()
(X.ix[train_len:, "IC_mean_12m"]/X.ix[train_len:, "IC_std_12m"]).plot.bar()
(X.ix[train_len:, "IC_mean_12m"]).plot.bar()
"""
注意到
"""    
    
y_hat_df.ix[train_len:, 'HSIGMA_orth'] = IC_mean_12m.ix[train_len:, 'HSIGMA_orth']
y_hat_df.loc[:, 'DIVIDENDRATE_orth'] = IC_mean_12m.ix[train_len:, 'DIVIDENDRATE_orth']
y_hat_df.loc[:, 'NETINCOME_su_orth'] = IC_mean_12m.ix[train_len:, 'NETINCOME_su_orth']
y_hat_df.loc[:, 'REVENUE_su_orth'] = IC_mean_12m.ix[train_len:, 'REVENUE_su_orth']

y_hat_df = y_hat_df.dropna(how="any")
factor_weights = y_hat_df.apply(lambda x: x/sum(np.abs(x)), axis=1)
np.abs(factor_weights).sum(1)


y_hat_df_lr.ix[train_len:, 'HSIGMA_orth'] = IC_mean_12m.ix[train_len:, 'HSIGMA_orth']
y_hat_df_lr.loc[:, 'DIVIDENDRATE_orth'] = IC_mean_12m.ix[train_len:, 'DIVIDENDRATE_orth']
y_hat_df_lr.loc[:, 'NETINCOME_su_orth'] = IC_mean_12m.ix[train_len:, 'NETINCOME_su_orth']
y_hat_df_lr.loc[:, 'REVENUE_su_orth'] = IC_mean_12m.ix[train_len:, 'REVENUE_su_orth']


y_hat_df_lr = y_hat_df_lr.dropna(how="any")
factor_weights = y_hat_df_lr.apply(lambda x: x/sum(np.abs(x)), axis=1)
np.abs(factor_weights).sum(1)

weighted_dates = factor_weights.index
weighted_dates = weighted_dates.strftime("%Y-%m-%d")




train_len = 36
test_len = 4
for i in range(train_len, len(X)-test_len):
    X_train = X.iloc[i-train_len: i]
    y_train = y.iloc[i-train_len: i]
    X_test = X.iloc[i:i+test_len]
    y_test = y.iloc[i:i+test_len]
    
    
    
    
    
def rolling_ml(train_len, test_len, X_train, y_train, model):
    
    return  
    
    
    
    











































































































































