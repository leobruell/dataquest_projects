#Predicting the Stock Market Guided Project 

import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

stocks = pd.read_csv('sphist.csv')
stocks['Date'] = pd.to_datetime(stocks['Date'])
stocks = stocks.sort_values(by='Date',ascending=True)

#Creating rolling average columns to use in predictions. 
stocks['5_day_avg'] = stocks['Close'].rolling(5, closed= "left").mean()
stocks['30_day_avg'] = stocks['Close'].rolling(30, closed= "left").mean()
stocks['5_day_sd'] = stocks['Close'].rolling(5, closed='left').std()
stocks['5_day_avg_vol'] = stocks['Volume'].rolling(5, closed= "left").mean()
stocks['high_low'] = stocks['High'] - stocks["Low"]
stocks['5_high_low_avg'] = stocks['high_low'].rolling(5, closed= "left").mean()


#Removing rows with null values, including the first 30 rows where rolling averages could not be completed. 
stocks.dropna(axis=0, inplace=True) 

#Dividing the data into testing and training sets
stocks.reset_index(drop=True, inplace=True)
train = stocks[ stocks['Date'] < datetime(2013, 1, 1)]
test = stocks[ stocks['Date'] >= datetime(2013,1,1)]

#First regression with 3 training variables
lin_reg = LinearRegression()
lin_reg.fit(train[['5_day_avg', '30_day_avg', '5_day_sd']], train['Close'])
predictions = lin_reg.predict(test[['5_day_avg', '30_day_avg', '5_day_sd']])
mae = mean_absolute_error(predictions, test['Close'])
rmse = mean_squared_error(predictions, test['Close'])**1/2
print(mae)
print(rmse)

#Additional regression with more training variables
lin_reg_2 = LinearRegression()
lin_reg_2.fit(train[['5_day_avg', '30_day_avg', '5_day_sd', '5_day_avg_vol', '5_high_low_avg']], train['Close'])
predictions_2 = lin_reg_2.predict(test[['5_day_avg', '30_day_avg', '5_day_sd', '5_day_avg_vol', '5_high_low_avg']])
mae_2 = mean_absolute_error(predictions_2, test['Close'])
rmse_2 = mean_squared_error(predictions_2, test['Close'])**1/2
print(mae_2)
print(rmse_2)
#Conclusion: Here, we see that adding these additional features to the model does not improve its accuracy. 
# This is likely because it is basically impossible to meaningfully predict a major stock index such as the S&P 500 based only on the data we used in this model. 
# If it was possible to make such predictions, people would buy and sell based on them and the market would correct itself (see the Efficient Market Hypothesis).

print('test')