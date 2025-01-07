import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2

from sklearn.preprocessing import MinMaxScaler

import os
import random

seed=1
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


cet_df = pd.read_csv("meantemp_monthly_totals.txt", delim_whitespace=True)[['year', 'Annual']]
pages2k_df = pd.read_csv("pages2k_processed.csv")
iso2k_df = pd.read_csv("iso2k_processed.csv")

paleo_df = pd.merge(pages2k_df, iso2k_df, on='year', how='outer')

def is_monotonically_increasing(series):
	series_no_nan = series.dropna()
	return series_no_nan.is_monotonic

years_range = range(1500, 1950)
paleo_df = paleo_df.loc[:, (paleo_df.loc[paleo_df['year'].isin(years_range)].notna().all())]

cols_to_remove = [col for col in paleo_df.columns if (col != 'year' and is_monotonically_increasing(paleo_df[col]))]
paleo_df = paleo_df.drop(columns=cols_to_remove)

print(paleo_df)

scaler = MinMaxScaler()
columns = [col for col in paleo_df.columns if col not in ['year']]
paleo_df[columns] = scaler.fit_transform(paleo_df[columns])


train_period = (1659, 1900)
test_period = (1900, 1949)


y_train = cet_df[(cet_df['year'] >= train_period[0]) & (cet_df['year'] <= train_period[1])]['Annual'].values #* 3600 * 24
y_test = cet_df[(cet_df['year'] >= test_period[0]) & (cet_df['year'] <= test_period[1])]['Annual'].values #* 3600 * 24


X_train = paleo_df[(paleo_df['year'] >= train_period[0]) & (paleo_df['year'] <= train_period[1])].drop(columns='year').values[:, 1:] #remove year
X_test = paleo_df[(paleo_df['year'] >= test_period[0]) & (paleo_df['year'] <= test_period[1])].drop(columns='year').values[:, 1:]



print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

if np.isnan(X_train).sum() == 0:
    print("X_train has no NaN values.")
else:
    print("X_train has NaN values.")

if np.isnan(X_test).sum() == 0:
    print("X_test has no NaN values.")
else:
    print("X_test has NaN values.")

early_stop = EarlyStopping(monitor='val_loss', patience=200, verbose=1, restore_best_weights=True)

model = Sequential([
    Dense(20, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.01)), 
    #BatchNormalization(), 
    Dropout(0.1), 

    #Dense(20, activation='relu', kernel_regularizer=l2(0.01)), 
    #Dropout(0.1),
    #Dense(20, activation='relu'), 
    #Dropout(0.1),
    Dense(1)
])



model.compile(optimizer=Adam(), loss='mse')

history = model.fit(X_train, y_train, epochs=1000, validation_data=(X_test, y_test), callbacks=[early_stop])


plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.show()


# Generate predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

from scipy.stats import linregress
print(linregress(train_predictions.ravel(), y_train))
print(linregress(test_predictions.ravel(), y_test))

# Create a figure and axis for plotting
plt.figure(figsize=(14, 6))
plt.title('Model Predictions vs. True Values')

# Plot true values for training and testing periods
plt.plot(cet_df[(cet_df['year'] >= train_period[0]) & (cet_df['year'] <= train_period[1])]['year'], y_train, 'b', label='True Train Values')
plt.plot(cet_df[(cet_df['year'] >= test_period[0]) & (cet_df['year'] <= test_period[1])]['year'], y_test, 'r', label='True Test Values')

# Plot predicted values for training and testing periods
plt.plot(cet_df[(cet_df['year'] >= train_period[0]) & (cet_df['year'] <= train_period[1])]['year'], train_predictions, 'b--', label='Predicted Train Values')
plt.plot(cet_df[(cet_df['year'] >= test_period[0]) & (cet_df['year'] <= test_period[1])]['year'], test_predictions, 'r--', label='Predicted Test Values')

# Adding labels, legend, and title
plt.xlabel('Year')
plt.ylabel('Summer Precipitation')
plt.legend()
plt.show()










'''
plt.plot(cet_df.year, cet_df.Annual.values*3600*24)

plt.show()'''
