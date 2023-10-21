import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import numpy as np
import math
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split

api_key = 'AQ48WEZTAJ8479HZ'

ts = TimeSeries(key=api_key, output_format='pandas')

symbol = 'NVDA'
start_date = '2017-01-01'
end_date = '2020-12-31'

data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
data = data[start_date:end_date]  # Filter data for the specified date range

df = data[['1. open', '2. high', '3. low', '4. close', '5. volume']]
df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

df['HL_PCT'] = (df['High']-df['Close'])/df['Close']*100
df['PCT_change']= (df['Close']-df['Open'])/df['Open']*100

df=df[['Close','HL_PCT','PCT_change','Volume']]

forcast_col = 'Close'
df.fillna(-99999, inplace=True)

forcast_out=int(math.ceil(0.1*len(df)))

df['label'] = df[forcast_col].shift(-forcast_out)
df.dropna(inplace=True)

X = np.array(df.drop(['label'],1))
Y = np.array(df['label'])
X = preprocessing.scale(X)

X_train , X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

clf = LinearRegression()
clf.fit(X_train, Y_train)

accuracy = clf.score(X_test, Y_test)
print(accuracy, forcast_out)