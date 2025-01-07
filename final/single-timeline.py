from functions import *

set_seed(1)



# Define region and resolution
x1, x2 = 40, 120  
y1, y2 = -10, 50
min_resolution = 10
paleo_df = prepare_paleo_df(x1,x2,y1,y2,min_resolution)


print(paleo_df)

#prcp_df = prepare_prcp_df("mean_seasonal_prcp_IMD")
prcp_df = prepare_prcp_df("1-nmi")




test_years = list(range(1975,1985))
valid_years = list(range(1950,1955))
train_years = [y for y in range(prcp_df['year'].min(),1995) if y not in test_years+valid_years]





y_train = prcp_df[prcp_df['year'].isin(train_years)]['summer_prcp'].values
y_test = prcp_df[prcp_df['year'].isin(test_years)]['summer_prcp'].values
y_val = prcp_df[prcp_df['year'].isin(valid_years)]['summer_prcp'].values

X_train = paleo_df[paleo_df['year'].isin(train_years)].drop(columns='year').values[:, 1:]
X_test = paleo_df[paleo_df['year'].isin(test_years)].drop(columns='year').values[:, 1:]
X_val = paleo_df[paleo_df['year'].isin(valid_years)].drop(columns='year').values[:, 1:]
X_extended = paleo_df[(paleo_df['year'] >= 1500) & (paleo_df['year'] < prcp_df['year'].min())].drop(columns='year').values[:, 1:]


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


early_stop = EarlyStopping(monitor='val_loss', patience=250, verbose=1, restore_best_weights=True)

model = Sequential([
    Dense(100, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l1(0.05)), 
    #BatchNormalization(), 
    #Dropout(0.2), 

    #Dense(20, activation='tanh', kernel_regularizer=l2(0.01)), 
    #Dropout(0.1),
    Dense(20, activation='linear'), 
    #Dropout(0.1),
    Dense(1)
])


loss_func = combined_loss(alpha=1., beta=1, gamma=2.)
model.compile(optimizer=Adam(), loss=loss_func)# loss=kge_loss)

history = model.fit(X_train, y_train, epochs=3000, validation_data=(X_val, y_val), callbacks=[early_stop])


plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.show()


# Generate predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)
extended_predictions = model.predict(X_extended)

from scipy.stats import linregress
print(linregress(train_predictions.ravel(), y_train))
print(linregress(test_predictions.ravel(), y_test))

# Create a figure and axis for plotting
plt.figure(figsize=(14, 6))
plt.title('Model Predictions vs. True Values')

# Plot true values for training and testing periods
plt.plot(prcp_df[prcp_df['year'].isin(train_years)]['year'], y_train, 'b', label='True Train Values')
plt.plot(prcp_df[prcp_df['year'].isin(test_years)]['year'], y_test, 'r', label='True Test Values')

# Plot predicted values for training and testing periods
plt.plot(prcp_df[prcp_df['year'].isin(train_years)]['year'], train_predictions, 'b--', label='Predicted Train Values')
plt.plot(prcp_df[prcp_df['year'].isin(test_years)]['year'], test_predictions, 'r--', label='Predicted Test Values')
plt.plot(paleo_df[(paleo_df['year'] >= 1500) & (paleo_df['year'] < prcp_df['year'].min())]['year'], extended_predictions, 'g--', label='Predicted Extended Values')

# Adding labels, legend, and title
plt.xlabel('Year')
plt.ylabel('Summer Precipitation')
plt.legend()
plt.show()










'''
plt.plot(prcp_df.year, prcp_df.Annual.values*3600*24)

plt.show()'''
