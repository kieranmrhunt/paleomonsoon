from functions import *
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.regularizers import l1, l2


# Function to build the neural network model
def build_model(input_shape, config):
	model = Sequential([
    Dense(config['L1_neurons'], activation=config['L1_activation'], input_shape=input_shape,
          kernel_regularizer=l1(config['L1_regularisation'])), 
    Dropout(config['dropout']), 
    Dense(config['L2_neurons'], activation='linear'), 
    Dense(1)
])
	loss_func = combined_loss(alpha=1., beta=1, gamma=2.)
	model.compile(optimizer=Adam(), loss=loss_func)
	return model

set_seed(1)

# Data Loading and Processing
prcp_df = prepare_prcp_df("mean_seasonal_prcp_IMD")
#prcp_df = prepare_prcp_df("3-nci")
paleo_df = prepare_paleo_df(40, 90, -10, 40, 10)

y0 = prcp_df['year'].min()
y1 = 1995

test_years = np.arange(1985,1995)
valid_years = np.arange(1980,1985)
train_years = [y for y in range(y0,y1) if y not in np.r_[test_years,valid_years]]

y_train = prcp_df[prcp_df['year'].isin(train_years)]['summer_prcp'].values
y_test = prcp_df[prcp_df['year'].isin(test_years)]['summer_prcp'].values
y_val = prcp_df[prcp_df['year'].isin(valid_years)]['summer_prcp'].values

X_train = paleo_df[paleo_df['year'].isin(train_years)].drop(columns='year').values[:, 1:]
X_test = paleo_df[paleo_df['year'].isin(test_years)].drop(columns='year').values[:, 1:]
X_val = paleo_df[paleo_df['year'].isin(valid_years)].drop(columns='year').values[:, 1:]
X_extended = paleo_df[(paleo_df['year'] >= 1500) & (paleo_df['year'] < y0)].drop(columns='year').values[:, 1:]


# Model Training
config = {
    'L1_neurons': 100, 
    'L2_neurons': 20, 
    'L1_activation': 'relu', 
    'dropout': 0.2, 
    'L1_regularisation': 0.01
}
model = build_model((X_train.shape[1],), config)
early_stop = EarlyStopping(monitor='val_loss', patience=250, verbose=1, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=3000, validation_data=(X_val, y_val), callbacks=[early_stop])

# Generate predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

fig = plt.figure(figsize=(15, 6))
gs = fig.add_gridspec(1, 3) 
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1:])

# Plot loss
ax1.plot(history.history['loss'], label='Training loss')
ax1.plot(history.history['val_loss'], label='Validation loss')

# Draw vertical line indicating epoch of best weights
best_epoch = np.argmin(history.history['val_loss'])
ax1.axvline(x=best_epoch, color='gray', linestyle='--', label='Best epoch')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss (standardised)')
ax1.set_xlim(0, len(history.history['loss']) - 1)  # Setting xlim for loss plot
ax1.legend()

# Plot observed data as one continuous grey line
all_years = prcp_df['year'].unique()
all_prcp = prcp_df['summer_prcp'].values
ax2.plot(all_years, all_prcp, 'grey', label='Observation')

# Calculate correlation coefficients
corr_train = pearsonr(y_train, train_predictions[:,0])[0]
corr_val = pearsonr(y_val, model.predict(X_val)[:,0])[0]
corr_test = pearsonr(y_test, test_predictions[:,0])[0]

# Plot individual predicted data segments
ax2.plot(train_years, train_predictions, 'b-', label=f'Predicted training (r={corr_train:.2f})')
ax2.plot(valid_years, model.predict(X_val), 'g-', label=f'Predicted validation (r={corr_val:.2f})')
ax2.plot(test_years, test_predictions, 'r-', label=f'Predicted test (r={corr_test:.2f})')

ax2.set_xlabel('Year')
ax2.set_ylabel('Standardisated summer precipitation')
ax2.set_xlim(1901, 1994)  # Setting xlim for time series plot
ax2.legend(loc='upper left', ncol=2)

plt.tight_layout()
fig.subplots_adjust(wspace=0.225)
plt.show()
