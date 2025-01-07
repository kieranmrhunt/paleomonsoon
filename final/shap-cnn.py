import shap
from functions import *
from tensorflow.keras.models import load_model
from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("shap/cnn/IMD-results.csv")
df['score'] = df['pcc'] - df['vd']/2
df_sorted = df.sort_values('score', ascending=False)


print(df_sorted)


model_data = df_sorted.iloc[0]
model_name = model_data.model_code

model_path = f'shap/cnn/models/{model_name}.h5'
model = load_model(model_path, custom_objects={ 'loss': combined_loss(alpha=1., beta=0.75, gamma=0.25)})

x1, x2 = 40, 95  
y1, y2 = -10, 40
min_resolution = 10
paleo_df, metadata = prepare_paleo_df(x1,x2,y1,y2,min_resolution, return_metadata=True)
print(paleo_df.columns)

X_full = paleo_df[(paleo_df['year'] >= 1500) & (paleo_df['year'] < 1995)].drop(columns='year').values[::5, 1:]
print(X_full.shape)

def model_predict(data):
    return model.predict(data).reshape((-1, 129*135))

e = shap.KernelExplainer(model_predict, X_full)
shap_values = np.array(e.shap_values(X_full)).squeeze()

print(shap_values)
print(np.shape(shap_values))

np.save("shap/cnn-shap-values", shap_values)

'''
importance = np.mean(np.abs(shap_values), axis=0)

print(metadata)

lons = metadata['geo_meanLon'].values[1:]
lats = metadata['geo_meanLat'].values[1:]
proxy_types = metadata['archiveType'].values[1:]
resolutions = metadata['resolution'].values[1:]

print(len(lons))


plt.scatter(lons, lats, c = importance, cmap=plt.cm.spectral)
plt.show()
'''

