import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import numpy as np

sales_train = pd.read_csv('.../sales_train_v2.csv')

#Creating new features, for months, years, days and monthyear:

dates = [i.replace("."," ").split() for i in sales_train["date"]]
months = [dates[i][1] for i in range(len(dates))]
years   = [dates[i][2] for i in range(len(dates))]
days   = [dates[i][0] for i in range(len(dates))]
monthyear = [months[i] + "." +years[i] for i in range(len(dates))]

#
sales_train["months"] = pd.DataFrame(months)
sales_train["years"]   = pd.DataFrame(years)
sales_train["days"]   = pd.DataFrame(days)
sales_train["monthyear"] = pd.DataFrame(monthyear)

#Creating month_counts data frame
month_counts = sales_train[["item_id", "monthyear", "item_cnt_day"]].groupby(["item_id", "monthyear"], as_index = False).sum()
month_counts["Sales"] = month_counts.item_cnt_day * sales_train.item_price

#density plot for item_price
sns.distplot(sales_train["item_price"])


#good transformations for NN / Select best transformation method for linear models

def log_tran(x):
    return np.log(1+x)

def sqrt_tran(x):
    return np.sqrt(x + 2/3)
