import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import r2_score

data = pd.read_csv("carinfo_after_pre_clean.csv")
select_columns = ['make', 'model', 'price', 'year', 'mileage', 'owner']
data = data.dropna(subset=select_columns)
df_selected = data[select_columns]
df_q5 = df_selected[(df_selected['price'] != 0) & (df_selected['year'] != 2025)]
df = df_q5

# calculate average price for each year each brand
brand_avg_prices = df.groupby(['year', 'make'])['price'].mean().unstack()
brand_avg_prices.fillna(method='ffill', inplace=True)
brand_avg_prices.fillna(method='bfill', inplace=True)

brands = brand_avg_prices.columns
feature_importance_df = pd.DataFrame(0, index=brands, columns=brands)
top_features_dict = {}  

# XGBoost feature importance 
for target_brand in brands:
    X = brand_avg_prices.drop(columns=[target_brand], errors='ignore')
    y = brand_avg_prices[target_brand]

    if y.isnull().all():
        continue

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    top_features = feature_importances.nlargest(3)
    top_features_dict[target_brand] = top_features


scaler = MinMaxScaler()
brand_avg_prices_scaled = scaler.fit_transform(brand_avg_prices)
brand_avg_prices_scaled = pd.DataFrame(brand_avg_prices_scaled, columns=brand_avg_prices.columns, index=brand_avg_prices.index)
top_features_df = pd.DataFrame.from_dict(top_features_dict, orient='index')
print("Top 3 important brands for each target brand:")
print(top_features_df)
metrics_df = pd.DataFrame(columns=['Brand', 'MAE', 'RMSE'])

top_features_df = top_features_df.fillna(0)
plt.figure(figsize=(12, 10))
sns.heatmap(top_features_df, cmap='viridis', annot=False, cbar_kws={'label': 'Feature Importance'})
plt.title('Brand Influence on Other Brands(Top 3)’ Prices')
plt.xlabel('Feature Brand')
plt.ylabel('Target Brand')
plt.savefig("top_3_brand_importance.png", dpi=300)



class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        output = self.fc(hn[-1])
        return output

def train_and_predict_for_brand(target_brand, top_features_df, brand_avg_prices_scaled, years=5, epochs=50):
    if target_brand not in top_features_df.index:
        print(f"Skipping {target_brand} as it's not in the top_features_df.")
        return None, None, None

    top_brands = top_features_df.loc[target_brand].index
    X, y = create_sequences(brand_avg_prices_scaled, target_brand, top_brands, years)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    
    input_size = X.shape[2]
    hidden_size = 64
    output_size = 1
    model = LSTMModel(input_size, hidden_size, output_size)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    loss_history = {}
    mae_history = []
    rmse_history = []
    metrics_results = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Target Brand: {target_brand}, Epoch {epoch}, Loss: {loss.item()}")
            if target_brand not in loss_history:
                loss_history[target_brand] = [] 
            loss_history[target_brand].append((epoch, loss.item()))

    
    target_brand_index = brand_avg_prices_scaled.columns.get_loc(target_brand)
    target_min = scaler.min_[target_brand_index]
    target_scale = scaler.scale_[target_brand_index]
    
    model.eval()
    with torch.no_grad():
        y_pred = model(X).numpy()
        y_actual = y.numpy()
        # y_pred_rescaled = y_pred * target_scale + target_min
        # y_actual_rescaled = y_actual * target_scale + target_min

        target_brand_index = brand_avg_prices_scaled.columns.get_loc(target_brand)

        y_pred_expanded = np.zeros((len(y_pred), len(brands)))
        y_actual_expanded = np.zeros((len(y_actual), len(brands)))

        y_pred_expanded[:, target_brand_index] = y_pred.flatten()
        y_actual_expanded[:, target_brand_index] = y_actual.flatten()

        y_pred_rescaled = scaler.inverse_transform(y_pred_expanded)[:, target_brand_index]
        y_actual_rescaled = scaler.inverse_transform(y_actual_expanded)[:, target_brand_index]


        # y_pred_rescaled = scaler.inverse_transform(y_pred)
        # y_actual_rescaled = scaler.inverse_transform(y_actual)

    
    mae = mean_absolute_error(y_actual_rescaled, y_pred_rescaled)
    mae_origin = mean_absolute_error(y_actual, y_pred)

    rmse = np.sqrt(mean_squared_error(y_actual_rescaled, y_pred_rescaled))
    rmse_origin = np.sqrt(mean_squared_error(y_actual, y_pred))
    mae_history.append(mae)
    rmse_history.append(rmse)
    print(f"Target Brand: {target_brand}, MAE: {mae}, RMSE: {rmse}")
    metrics_results.append({'Brand': target_brand, 'MAE': mae, 'RMSE': rmse})

    return y_actual_rescaled, y_pred_rescaled, target_brand, loss_history,mae_history,rmse_history,metrics_results

def create_sequences(data, target_brand, top_brands, years):
    X, y = [], []
    for i in range(years, len(data)):
        X.append(data.iloc[i - years:i][top_brands].values)
        y.append(data.iloc[i, data.columns.get_loc(target_brand)])
    return np.array(X), np.array(y).reshape(-1, 1)



loss_all_brand = []
results = {}
metrics_final = []

for target_brand in top_features_df.index:
    y_actual_rescaled, y_pred_rescaled, target_brand, loss_history,mae_history,rmse_history,metrics_results = train_and_predict_for_brand(target_brand, top_features_df, brand_avg_prices_scaled)
    if y_actual_rescaled is not None:
        results[target_brand] = (y_actual_rescaled, y_pred_rescaled)
    loss_all_brand.append(loss_history)
    metrics_final.append({'Brand': target_brand, 'MAE': mae_history, 'RMSE': rmse_history})

metrics_df = pd.DataFrame(metrics_final)

metrics_df['MAE'] = metrics_df['MAE'].apply(lambda x: float(str(x).strip('[]')))
metrics_df['RMSE'] = metrics_df['RMSE'].apply(lambda x: float(str(x).strip('[]')))
metrics_df.to_csv('brand_metrics_results.csv', index=False)



# save brands
for target_brand, (y_actual_rescaled, y_pred_rescaled) in results.items():
    plt.figure(figsize=(10, 6))
    plt.plot(y_actual_rescaled, label='Actual Price')
    plt.plot(y_pred_rescaled, label='Predicted Price')
    plt.title(f'LSTM Prediction for {target_brand} Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(f"{target_brand}_prediction.png", dpi=300)
    plt.close()

r2 = r2_score(y_actual_rescaled, y_pred_rescaled)
print(r2)
plt.figure(figsize=(12, 8))
# print(loss_history)

plt.figure(figsize=(12, 8))

for brand_data in loss_all_brand:
    for brand, losses in brand_data.items():
        epochs, loss_values = zip(*losses)  
        plt.plot(epochs, loss_values, label=brand)  

plt.yscale("log")

plt.title("Loss over Epochs for All Brands")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig("all_brands_loss_curve.png", dpi=300)


