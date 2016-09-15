# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 15:28:39 2016

@author: mlamp
"""

import pandas as pd
import numpy as np
import re

dirty_path_lib = "/var/lib/spark/data/"
clean_path_lib = "/var/lib/spark/data_clean/"
USER_ORDERS = "xboss_user_orders"
PAY_PLANS = "xboss_user_pay_plans"

'''
读取数据
'''
def load_data(prefix,filename):
    schema = pd.read_csv(prefix+filename+".schema","\t",
                         names=["name","dtype","blank"],na_values=["null"])
    names = [x.strip() for x in schema.name]
    df = pd.read_csv(prefix+filename,sep = "\001",
                     names = names,na_values=["null"])
    return df

'''
年化收益率符号
'''
def get_symbol(x,pattern):
    m = pattern.search(x)
    return m.group(2)

'''
年化收益率
'''
def get_rate(x,pattern):
    m = pattern.match(x)
    var1 = m.group(1)
    var2 = m.group(3)
    symbol = m.group(2)    
    
    if var2 == "" or var2 == "X":
        return float(var1)
    else:
        if symbol == "+":
            return float(var1)+float(var2)
        elif symbol == "~":
            return np.mean([float(var1),float(var2)])
'''
清洗年化收益率
'''
def get_annualized_rate(annualized_rate):
    pattern = re.compile(r"([-\d.]*)%*([\+~]*)([X\d.]*)%*")
    return annualized_rate.map(lambda x: get_rate(x,pattern))  
'''
重复购买率
总共：26个月
'''
def re_buy_rate(df):
    from dateutil.relativedelta import relativedelta
    import datetime
    max_time = df.order_time.max(axis=0)    
    train_time = pd.to_datetime("2014-08-01")
    print("年 月份 复购率 复购人数 新用户 总人数 当月单量 金额 人均购买")
    num = 0
    while train_time<max_time:
        try:
            label_time = datetime.datetime(train_time.year,train_time.month+1,train_time.day)
        except ValueError:
            label_time = datetime.datetime(train_time.year+1,1,train_time.day)
        df_train = df[df.order_time<train_time]
        df_label = df[df.order_time.between(train_time,label_time)]
        df_train_u_id_uni = df_train.u_id.unique()
        df_label_u_id_uni = df_label.u_id.unique()
        intersection = np.intersect1d(df_train_u_id_uni,df_label_u_id_uni)
        total_money = df_label.order_amount.sum(axis=0)
        print(train_time.year,
              "%2d"%train_time.month,
              "%6f"%(len(intersection)/len(df_train_u_id_uni)),
              "%5d"%len(intersection),
              "%5d"%(len(df_label.u_id.unique())-len(intersection)),
              "%7d"%len(df_label.u_id.unique()),
              "%7d"%len(df_label.u_id),
              "%15d"%total_money,
              "%10d"%(total_money/len(df_label.u_id)))
        train_time = train_time+relativedelta(months=1)

def joint(X):
    return str(X[0])+"-"+str(X[1])+"-"+"01"
      
def unify_time(df):
    import datetime
    order_time = pd.to_datetime(df.order_time)
    df["year"],df["month"] = order_time.dt.year,order_time.dt.month
    df["unify_time"] = df[["year","month"]].apply(joint,axis=1)
    df.unify_time = pd.to_datetime(df.unify_time)
    df.drop(["year","month"],inplace=True)
    return df
#%%    
def time_var(time,var=None):
    import datetime
    if var == None:
        return pd.to_datetime("2000-01-01")
    elif var>0:
        for i in range(1,var+1):
            try:
                time = datetime.datetime(time.year,time.month+1,time.day)
            except:
                time = datetime.datetime(time.year+1,1,time.day)
        return time
    elif var<=0:
        for i in range(1,1-var):
            try:
                time = datetime.datetime(time.year,time.month-1,time.day)
            except:
                time = datetime.datetime(time.year-1,12,time.day)
        return time

'''
提取数据

train_months = None ->使用全部历史数据
train_months = Integer -> 使用最近的N个月

'''    
def extract_data(df,train_months=None,test_months=None):
    if test_months == None:
        raise ValueError("test_months must be given.")
    from datetime import datetime as dt
    from datetime import timedelta
    one_day = timedelta(days=1)
    
    max_time = pd.to_datetime(df.order_time.max(axis=0))
    min_time = pd.to_datetime(df.order_time.min(axis=0))
    
    start_time = dt(min_time.year,min_time.month,1)
    
    if train_months == None:
        sep_time = time_var(start_time,var=1)
    else:        
        sep_time = time_var(start_time,var=train_months)
    
    df.order_time = pd.to_datetime(df.order_time)
    data_ls = []
    sep_ls = []
    num = 0
    while sep_time < max_time:
        target_end_time = time_var(sep_time,var=test_months)
        
        if train_months == None:
            fea_start_time = start_time
        else:
            fea_start_time = time_var(sep_time,var=-train_months)
        
        feature_data = df[df.order_time.between(fea_start_time,sep_time-one_day)]
        target_data = df[df.order_time.between(sep_time,target_end_time-one_day)]
        
        sep_ls.append(sep_time)
        feature_data["sep_time"] = sep_time
        feature_data["num"] = num
        
        u_id_tg = target_data.u_id.unique()
        u_id_ft = feature_data.u_id
        
        feature_data["target"] = np.in1d(u_id_ft,u_id_tg)        
        feature_data.target = feature_data.target.astype(np.int8)
        
        data_ls.append(feature_data)
        
        sep_time =  time_var(sep_time,var=test_months)
        num += 1
        
    return pd.concat(data_ls,axis=0).reset_index(drop=True),sep_ls
    
#%%
'''
数据类型转换
'''
def time_str2date(data,col_time):  
    temp = [pd.to_datetime(data[col]) for col in col_time]
    return pd.concat(temp,axis=1)
    
'''
时间--->相对时间
'''   
def time_abs2rlt(data,col_time):
    def trans(x):
        return pd.Series(x.values.astype("timedelta64[D]")/np.timedelta64(30,"D"))
    temp = [trans(data["sep_time"]-data[c]) for c in col_time]   
    return pd.concat(temp,axis=1)

'''
类别变量预处理
'''
def onehot(data,col_cls):
    onehot_vars = []
    for c in col_cls:
        top = data[c].value_counts().nlargest(100).index
        if data[c].dtypes == object:
            data[c] = data[c].map(lambda x: x if x in top else "unicode")
        else:
            data[c] = data[c].map(lambda x: x if x in top else -11)
        temp = pd.get_dummies(data[c],prefix=c,prefix_sep='_')
        onehot_vars.append(temp)
    onehot_vars = pd.concat(onehot_vars,axis=1)
    data.drop(col_cls,axis=1,inplace=True)
    data = pd.concat([data,onehot_vars],axis=1)
    return data,onehot_vars.columns

'''
提取
'''           
def extract_feature(data,col_num,col_time,onehot_name,is_dcp=True):
    import gc
    nums = data["num"].unique()    
    
    #abs_time->rlt_time
    print("absolute_time-->relaive time")
    data[col_time] = time_abs2rlt(data,col_time)    
    data[col_time] = data[col_time].fillna(data["sep_time"],axis=0)
    
    grouped = data.groupby(["u_id","num"],sort=False)
    #num+time
    print("extract num and time feature....")
    feature = grouped[col_num+col_time].agg([np.sum,np.mean,np.std,np.max,np.min,np.median])
    temp_cols = feature.columns.tolist()
    columns = [l[0]+"_"+l[1] for l in temp_cols]
    feature.columns = columns
    #cls
    print("combine one hot feature....")
    temp_cls = grouped[onehot_name].sum()
    feature = pd.concat([feature,temp_cls],axis=1)
    #target
    print("extract target......")
    temp_target = grouped["target"].mean()
    feature = pd.concat([feature,temp_target],axis=1)
    
    feature.reset_index(inplace=True)
    
    print("output.....")
    if is_dcp == True:  
        feature_names,target_name = feature.columns[2:-1],"target"
        
        feature_ls = []
        target_ls = []
        
        for i in nums:
            print(i,end=" ")
            temp = (feature["num"]==i)
            feature_ls.append(feature[temp][feature_names])
            target_ls.append(feature[temp][target_name])
        
        return feature_ls,target_ls 
    
    else:
        feature_names = feature.columns[1:-1]
        target_name = "target" 
        return feature[feature_names],feature[target_name]

#%%
def train(train_feature,train_target,time,test_feature,test_target,cheat_ID=None):
    import xgboost as xgb
    from sklearn.cross_validation import train_test_split    
    
    X_train,X_test,y_train,y_test = train_test_split(train_feature,
                                                     train_target,
                                                     test_size=0.1,
                                                     random_state=2016)    
    xlf = xgb.XGBClassifier(max_depth=4,
                            learning_rate=0.1,
                            n_estimators=100,
                            silent=True,
                            subsample=0.8,
                            colsample_bytree=0.8)  
    eval_set =([X_train,y_train],[X_test,y_test])
    #TRAIN
    xlf.fit(X_train,y_train,
                    eval_metric="auc",
                    verbose=False,
                    eval_set=eval_set,
                    early_stopping_rounds=5)
    #PREDICT                    
    pred = xlf.predict(test_feature)
    pred_proba = xlf.predict_proba(test_feature)[:,1]
    #TEST_BASE
    test_base = test_target.sum()/len(test_target)
    #TRAIN_BASE
    train_num = len(X_train)
    train_base = y_train.sum()/len(y_train)    
    
    #INDICATOR
    from sklearn.metrics import f1_score
    f1 = f1_score(test_target,pred)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(test_target,pred_proba)
    
    time = str(time.year)+"-"+str(time.month)
    print("%10s\t%.10f\t%.10f\t%5d\t%.10f\t%.10f"%(time,test_base,train_base,train_num,                                                                                    
                                                                                     f1,
                                                                                    auc))                                                                             
    return xlf.feature_importances_

#%%
if __name__ == "__main__":
    user_order_cls = ["levels","project_name","pay_channel","sale_mode",
                      "order_type","can_transfer","buy_source","pay_mode",
                      "has_plan","is_stop"]
    user_order_num = ["order_share","order_amount","interest","annualized_rate",
                      "ann_rev","transfer_fee_rate","transfer_fee","transfer_discount"]               
    user_order_time = ["order_time","repay_date","start_interest_date","created_at",
                      "updated_at"]
    #LOAD
    df_user_order = load_data(dirty_path_lib,USER_ORDERS)
    #PRE_PROCESSING
    df_user_order.annualized_rate = get_annualized_rate(df_user_order.annualized_rate)
    df_user_order,onehot_names = onehot(df_user_order,user_order_cls)
    df_user_order[user_order_time] = time_str2date(df_user_order,user_order_time)
    
    feature_ls,target_ls = extract_feature(data,user_order_num,
                                           user_order_time,onehot_names) 
    
    ft_imp = []
    print("\ttime\ttest_base\ttrain_base\ttrain_num\tf1\tauc")
    for f_train,t_train,s,f_test,t_test in zip(feature_ls[:-1],target_ls[:-1],
                                               sep_ls[1:],feature_ls[1:],target_ls[1:]):
        temp = train(f_train,t_train,s,f_test,t_test)
        ft_imp.append(temp)
        
        
    
    

    


