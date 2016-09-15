# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 10:16:56 2016

@author: mlamp
"""

import pyspark as ps
import pandas
from datetime import datetime

datetime.strptime("2016-07-07 23:32:31","%Y-%m-%d %H:%M:%S")

#显示表的列名
cols = sqlCtx.sql("show columns from sdata.xboss_user_orders")
df = sqlCtx.sql("select count(distinct u_id) from sdata.xboss_user_orders")

'''
xboss_user_orders: count(u_id) min:2 max:432 under u_id>1
'''
df = sqlCtx.sql("select count(u_id) from sdata.xboss_user_orders group by u_id having count(u_id)>1")
df.describe().show()

'''
显示多次出现的数据
'''
users = sqlCtx.sql("select u_id,count(u_id) as count from sdata.xboss_user_orders group by u_id having count(u_id)>1")
total_users_info = sqlCtx.sql("select * from sdata.xboss_user_orders")
users_dup = users.join(total_users_info,"u_id",how="left_outer")
'''
时间窗口滑动
'''
total_users_info = sqlCtx.sql("select * from sdata.xboss_user_orders")
##select是用来生成一个新的dataframe
##cast转换数据类型
from pyspark.sql.functions import year,min,mean

num_days = 30

#先把想要的月份找出来
users_id_time = total_users_info.select(total_users_info.u_id,
                                        total_users_info.order_time.cast("timestamp").alias("time"))









users_id_time = total_users_info.select(total_users_info.u_id,
                                        total_users_info.order_time.cast("timestamp").alias("time"),
                                        year(total_users_info.order_time).alias("year"))
users_id_time.filter(users_id_time.year>2015).show()



def is_satisfy(x):
    date = datetime.strptime(x.order_time,"%Y-%m-%d %H:%M:%S")
    if date.year>2015:
        return True
    else:
        return False
















from pyspark.sql.functions import collect_list
total_users_info = sqlCtx.sql("select * from sdata.xboss_user_orders")
list_dup = total_users_info.select(collect_list("u_id"))







df = sqlCtx.sql("select u_id,first(order_time) from sdata.xboss_user_orders group by u_id having count(u_id)>1")
df = sqlCtx.sql("select u_id,order_time from sdata.xboss_user_orders having u_id in (select u_id from sdata.xboss_user_orders group by u_id having count(u_id)>1)")