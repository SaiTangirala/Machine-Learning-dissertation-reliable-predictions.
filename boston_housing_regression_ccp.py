#  regression for Boston dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping
from google.colab import files
import io
from scipy.sparse import issparse

class EpochProgressCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        total_epochs = self.params['epochs']
        remaining_epochs = total_epochs - (epoch + 1)
        print(f"Epoch {epoch + 1}/{total_epochs} - Remaining epochs: {remaining_epochs}")


uploaded = files.upload()
file_name = list(uploaded.keys())[0]
data = pd.read_csv(io.StringIO(uploaded[file_name].decode('utf-8')))

print("First few rows of the dataset:")
print(data.head())
print("\nColumn names in the dataset:")
print(data.columns)

target_column = 'MEDV'  
categorical_features = ['TOWN', 'TOWN.1', 'TRACT', 'CHAS']
numeric_features = ['LON', 'LAT', 'CMEDV', 'CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

if target_column not in data.columns:
    raise KeyError(f"'{target_column}' not found in dataset columns")

X = data[categorical_features + numeric_features]
y = data[target_column]

X = X.dropna()
y = y[X.index]  

for feature in categorical_features + numeric_features:
    if feature not in X.columns:
        raise KeyError(f"'{feature}' not found in dataset columns")
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_features)
    ])

X_transformed = preprocessor.fit_transform(X)

if issparse(X_transformed):
    X_transformed = X_transformed.toarray()

def create_model(input_dim, optimizer='adam', learn_rate=0.001):
    model = Sequential([
        Dense(128, input_dim=input_dim, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1) 
    ])
    model.compile(optimizer=Adam(learning_rate=learn_rate), loss='mean_squared_error')
    return model


k = 5 
kf = KFold(n_splits=k, shuffle=True, random_state=42)


alpha_values = np.linspace(0.1, 0.9, 9)

all_error_rates = []


for alpha in alpha_values:
    print(f"Evaluating significance level: {alpha}")
    error_rate_list = []
    for fold, (train_index, test_index) in enumerate(kf.split(X_transformed)):
        print(f"Processing Fold {fold + 1}/{k}")
        X_train, X_test = X_transformed[train_index], X_transformed[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        x_train_proper, x_calibrate, y_train_proper, y_calibrate = train_test_split(X_train, y_train, test_size=0.2)
        model = create_model(X_train.shape[1])
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(
            x_train_proper, y_train_proper,
            epochs=20,
            validation_data=(x_calibrate, y_calibrate),
            callbacks=[EpochProgressCallback(), early_stopping],
            verbose=0 
        )

      
        y_pred_calibrate = model.predict(x_calibrate).flatten()
        nonconformity_scores = np.abs(y_pred_calibrate - y_calibrate)
        nonconformity_scores = sorted(nonconformity_scores)
        quant = int(np.ceil((1 - alpha) * (len(x_calibrate) + 1)))
        quantile = nonconformity_scores[quant]

      
        y_pred_test = model.predict(X_test)

        predictions = []
        for i in range(len(y_pred_test)):
           
            lower_bounds = y_pred_test[i] - quantile
            upper_bounds = y_pred_test[i] + quantile
            predictions.append([lower_bounds.item(), upper_bounds.item()])
        errors = 0
        y_test = y_test.reset_index(drop=True) 
        for i in range(len(predictions)):
            if y_test[i] < predictions[i][0] or y_test[i] > predictions[i][1]:
                errors += 1
        error_rate = errors / len(predictions)
        error_rate_list.append(error_rate)

    avg_error_rate = np.mean(error_rate_list)
    all_error_rates.append(avg_error_rate)

plt.figure(figsize=(10, 6))
plt.plot(alpha_values, all_error_rates, marker='o', linestyle='-', color='b')
plt.xlabel('Significance Level')
plt.ylabel('Error Rate')
plt.title('Calibration Graph: Error Rate vs Significance Level')
plt.grid(True)
plt.show()
