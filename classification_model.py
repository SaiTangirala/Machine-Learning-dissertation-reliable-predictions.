# classification code

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from google.colab import files
import io
import matplotlib.pyplot as plt

class EnhancedLungCancerNN(nn.Module):
    def __init__(self, input_dim):
        super(EnhancedLungCancerNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5) 

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc4(x)) 
        return x
def preprocess_and_prepare_data(file):
    data = pd.read_csv(io.StringIO(file.decode('utf-8')))
    data = data.dropna()
    data['GENDER'] = data['GENDER'].map({'M': 1, 'F': 0})
    data['LUNG_CANCER'] = data['LUNG_CANCER'].map({'YES': 1, 'NO': 0})
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

  
    X_train, X_temp, Y_train, Y_temp = train_test_split(X_resampled, y_resampled, test_size=0.4, random_state=42)
    X_val, X_calib, Y_val, Y_calib = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

    return (torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32)), \
           (torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.float32)), \
           (torch.tensor(X_calib, dtype=torch.float32), torch.tensor(Y_calib, dtype=torch.float32)), \
           scaler


def train_model(model, train_loader, val_loader, epochs=100, learning_rate=0.001, patience=10):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_loss = evaluate_model(model, val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    return train_losses, val_losses

def evaluate_model(model, val_loader):
    model.eval()
    criterion = nn.BCELoss()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    return val_loss / len(val_loader)

def calculate_conformity_scores(probs, labels):
    scores = []
    for i in range(len(probs)):
        prob = probs[i]
        label = int(labels[i])
        score = -np.log(prob) if label == 1 else -np.log(1 - prob)
        scores.append(score)
    return np.array(scores)

def cross_conformal_prediction(X_calib, Y_calib, model, alpha=0.05):
    all_predictions = []
    all_conformity_scores = []

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, test_idx in kf.split(X_calib):
        X_train_fold, X_test_fold = X_calib[train_idx], X_calib[test_idx]
        Y_train_fold, Y_test_fold = Y_calib[train_idx], Y_calib[test_idx]
        X_train_tensor = torch.tensor(X_train_fold, dtype=torch.float32)
        Y_train_tensor = torch.tensor(Y_train_fold, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_fold, dtype=torch.float32)
        fold_model = EnhancedLungCancerNN(X_train_tensor.shape[1])
        train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_dataset = TensorDataset(X_test_tensor, torch.tensor(Y_test_fold, dtype=torch.float32))
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        train_model(fold_model, train_loader, val_loader, epochs=50)

        
        fold_model.eval()
        with torch.no_grad():
            probs = fold_model(X_test_tensor).squeeze().numpy()

        
        scores = calculate_conformity_scores(probs, Y_test_fold)
        all_predictions.append(probs)
        all_conformity_scores.append(scores)

    all_predictions = np.concatenate(all_predictions)
    all_conformity_scores = np.concatenate(all_conformity_scores)
    threshold = np.percentile(all_conformity_scores, 100 * (1 - alpha))

   
    p_values = np.array([
        np.mean(all_conformity_scores >= score)
        for score in all_conformity_scores
    ])

    return all_predictions, all_conformity_scores, threshold, p_values

def plot_loss_curve(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
def print_error_rate_vs_significance(p_values, Y_calib, significance_levels=np.linspace(0.01, 0.1, 10)):
    error_rates = []

    for alpha in significance_levels:
        threshold = np.percentile(p_values, 100 * (1 - alpha))
        predictions = (p_values >= threshold).astype(int)
        error_rate = 1 - accuracy_score(Y_calib, predictions)
        error_rates.append(error_rate)

    for alpha, error_rate in zip(significance_levels, error_rates):
        print(f"Significance Level: {alpha:.2f}, Error Rate: {error_rate:.4f}")

uploaded = files.upload()
file_name = list(uploaded.keys())[0]
(X_train_tensor, Y_train_tensor), (X_val_tensor, Y_val_tensor), (X_calib_tensor, Y_calib_tensor), scaler = preprocess_and_prepare_data(uploaded[file_name])

final_model = EnhancedLungCancerNN(X_train_tensor.shape[1])
train_loader = DataLoader(TensorDataset(X_train_tensor, Y_train_tensor), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, Y_val_tensor), batch_size=32, shuffle=False)
train_losses, val_losses = train_model(final_model, train_loader, val_loader, epochs=100, learning_rate=0.001, patience=10)

plot_loss_curve(train_losses, val_losses)
alpha = 0.02
all_predictions, all_conformity_scores, threshold, p_values = cross_conformal_prediction(X_calib_tensor.numpy(), Y_calib_tensor.numpy(), final_model, alpha=alpha)
print_error_rate_vs_significance(p_values, Y_calib_tensor.numpy(), significance_levels=np.linspace(0.01, 0.1, 10))
