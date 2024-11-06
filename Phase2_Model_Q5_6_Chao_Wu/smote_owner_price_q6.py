from imblearn.over_sampling import SMOTENC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("carinfo_after_pre_clean.csv")
select_columns = ['make', 'model', 'price', 'year', 'mileage', 'owner']
data = data.dropna(subset=select_columns)
df_selected = data[select_columns]
df_q5 = df_selected[(df_selected['price'] != 0) & (df_selected['year'] != 2025)]

data_4_5 = df_q5[df_q5['owner'].isin([4, 5])]

X = data_4_5[['make', 'model', 'price', 'year', 'mileage', 'owner']]
y = data_4_5['owner']  

smote_nc = SMOTENC(categorical_features=[0, 1, 3, 5], random_state=42)

X_resampled, y_resampled = smote_nc.fit_resample(X, y)

X_resampled['year'] = X_resampled['year'].round().astype(int)
X_resampled['owner'] = X_resampled['owner'].round().astype(int)

generated_data = pd.DataFrame(X_resampled, columns=['make', 'model', 'price', 'year', 'mileage', 'owner'])
print(generated_data.head())
brand_counts = generated_data['make'].value_counts()
origin_brand_counts = df_q5['make'].value_counts()

print("smote data‘ brand count：")
print(brand_counts)

selected_brands = ["Subaru", "Ford", "Chevrolet", "Porsche", "Ram", "Tesla"]
filtered_data = generated_data[generated_data['make'].isin(selected_brands)]

filtered_data.to_csv('lower_rmse_sys.csv', index=False)


for brand in selected_brands:
    print(f"\nStatistics for {brand}:")
    print(filtered_data[filtered_data['make'] == brand].describe())


print("real data‘ brand count：")
print(origin_brand_counts)

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


features = [ 'year', 'mileage', 'owner','make']  
target = 'price'  
df_combined = pd.concat([df_q5, generated_data], ignore_index=True)

X_combined = df_combined[['year', 'mileage', 'owner']]

y_combined = df_combined[target].astype(float)

X_real = df_q5[['year', 'mileage', 'owner']]
y_real = df_q5[target].astype(float)
print(generated_data['owner'].value_counts())


X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)
X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(X_real, y_real, test_size=0.2, random_state=42)

X_train_combined = torch.tensor(X_train_combined.values, dtype=torch.float32)
y_train_combined = torch.tensor(y_train_combined.values, dtype=torch.float32).view(-1, 1)
X_test_combined = torch.tensor(X_test_combined.values, dtype=torch.float32)
y_test_combined = torch.tensor(y_test_combined.values, dtype=torch.float32).view(-1, 1)

X_train_real = torch.tensor(X_train_real.values, dtype=torch.float32)
y_train_real = torch.tensor(y_train_real.values, dtype=torch.float32).view(-1, 1)
X_test_real = torch.tensor(X_test_real.values, dtype=torch.float32)
y_test_real = torch.tensor(y_test_real.values, dtype=torch.float32).view(-1, 1)

base_model_rf = RandomForestRegressor(n_estimators=50, random_state=42)
base_model_lr = LinearRegression()

def train_base_models(X_train, y_train):
    base_model_rf.fit(X_train, y_train)
    base_model_lr.fit(X_train, y_train)
    
    rf_preds = base_model_rf.predict(X_train)
    lr_preds = base_model_lr.predict(X_train)
    
    return np.vstack([rf_preds, lr_preds]).T

class MetaModel(nn.Module):
    def __init__(self):
        super(MetaModel, self).__init__()
        self.linear = nn.Linear(2, 1) 
    def forward(self, x):
        return self.linear(x)

def train_meta_model(meta_model, train_data, y_train, test_data, y_test):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(meta_model.parameters(), lr=0.01)
    
    for epoch in range(1000):
        meta_model.train()
        optimizer.zero_grad()
        outputs = meta_model(train_data)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    meta_model.eval()
    with torch.no_grad():
        predictions = meta_model(test_data)
        mse = criterion(predictions, y_test).item()
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test.numpy(), predictions.numpy())

    return mse, rmse, mae

train_preds_combined = train_base_models(X_train_combined, y_train_combined.numpy().flatten())
test_preds_combined = train_base_models(X_test_combined, y_test_combined.numpy().flatten())
train_data_combined = torch.tensor(train_preds_combined, dtype=torch.float32)
test_data_combined = torch.tensor(test_preds_combined, dtype=torch.float32)

meta_model_combined = MetaModel()
mse_combined, rmse_combined, mae_combined = train_meta_model(meta_model_combined, train_data_combined, y_train_combined, test_data_combined, y_test_combined)

print(f"smote data overall RMSE: {rmse_combined:.4f}, MAE: {mae_combined:.4f}")

# train stack model using real data
train_preds_real = train_base_models(X_train_real, y_train_real.numpy().flatten())
test_preds_real = train_base_models(X_test_real, y_test_real.numpy().flatten())
train_data_real = torch.tensor(train_preds_real, dtype=torch.float32)
test_data_real = torch.tensor(test_preds_real, dtype=torch.float32)

meta_model_real = MetaModel()
mse_real, rmse_real, mae_real = train_meta_model(meta_model_real, train_data_real, y_train_real, test_data_real, y_test_real)

print(f"real data overall RMSE: {rmse_real:.4f}, MAE: {mae_real:.4f}")






from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import torch
import numpy as np

# encode make as integer
label_encoder = LabelEncoder()
df_combined['make'] = label_encoder.fit_transform(df_combined['make'])
df_q5['make'] = label_encoder.transform(df_q5['make'])

def evaluate_by_brand(df_combined, df_q5, brand_column='make'):
    brand_results = defaultdict(dict)  
    
    brands = df_combined[brand_column].unique()
    
    for brand in brands:
        brand_combined = df_combined[df_combined[brand_column] == brand]
        brand_real = df_q5[df_q5[brand_column] == brand]
        
        if len(brand_real) < 10 or len(brand_combined) < 10:
            print(f"brand {label_encoder.inverse_transform([brand])[0]} not enough data points, skipped")
            continue
        
        X_combined = brand_combined[features].astype(float)
        y_combined = brand_combined[target].astype(float)
        X_real = brand_real[features].astype(float)
        y_real = brand_real[target].astype(float)
        
        X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(
            X_combined, y_combined, test_size=0.2, random_state=42)
        X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
            X_real, y_real, test_size=0.2, random_state=42)
        
        X_train_combined = torch.tensor(X_train_combined.values, dtype=torch.float32)
        y_train_combined = torch.tensor(y_train_combined.values, dtype=torch.float32).view(-1, 1)
        X_test_combined = torch.tensor(X_test_combined.values, dtype=torch.float32)
        y_test_combined = torch.tensor(y_test_combined.values, dtype=torch.float32).view(-1, 1)

        X_train_real = torch.tensor(X_train_real.values, dtype=torch.float32)
        y_train_real = torch.tensor(y_train_real.values, dtype=torch.float32).view(-1, 1)
        X_test_real = torch.tensor(X_test_real.values, dtype=torch.float32)
        y_test_real = torch.tensor(y_test_real.values, dtype=torch.float32).view(-1, 1)
        
        train_preds_combined = train_base_models(X_train_combined, y_train_combined.numpy().flatten())
        test_preds_combined = train_base_models(X_test_combined, y_test_combined.numpy().flatten())
        train_preds_real = train_base_models(X_train_real, y_train_real.numpy().flatten())
        test_preds_real = train_base_models(X_test_real, y_test_real.numpy().flatten())
        
        train_data_combined = torch.tensor(train_preds_combined, dtype=torch.float32)
        test_data_combined = torch.tensor(test_preds_combined, dtype=torch.float32)
        train_data_real = torch.tensor(train_preds_real, dtype=torch.float32)
        test_data_real = torch.tensor(test_preds_real, dtype=torch.float32)
        
        meta_model_combined = MetaModel()
        mse_combined, rmse_combined, mae_combined = train_meta_model(
            meta_model_combined, train_data_combined, y_train_combined, test_data_combined, y_test_combined)

        meta_model_real = MetaModel()
        mse_real, rmse_real, mae_real = train_meta_model(
            meta_model_real, train_data_real, y_train_real, test_data_real, y_test_real)
        
        brand_name = label_encoder.inverse_transform([brand])[0]
        brand_results[brand_name]['synthetic'] = {'MSE': mse_combined, 'RMSE': rmse_combined, 'MAE': mae_combined}
        brand_results[brand_name]['real'] = {'MSE': mse_real, 'RMSE': rmse_real, 'MAE': mae_real}
        
        print(f"{brand_name} - smote data: MSE={mse_combined:.4f}, RMSE={rmse_combined:.4f}, MAE={mae_combined:.4f}")
        print(f"{brand_name} - real data: MSE={mse_real:.4f}, RMSE={rmse_real:.4f}, MAE={mae_real:.4f}")
        
    return brand_results

results = evaluate_by_brand(df_combined, df_q5)



# group by owner
owner_groups_combined = {owner: df_combined[df_combined['owner'] == owner] for owner in df_combined['owner'].unique()}
owner_groups_real = {owner: df_q5[df_q5['owner'] == owner] for owner in df_q5['owner'].unique()}

def evaluate_by_owner(owner_groups_combined, owner_groups_real):
    owner_results = {}

    for owner, data_combined in owner_groups_combined.items():
        data_real = owner_groups_real.get(owner)
        
        if data_real is None or len(data_real) < 10 or len(data_combined) < 10:
            print(f"Owner {owner} - not enough data points, skipped")
            continue
        
        X_combined = data_combined[['year', 'mileage', 'make']].astype(float)
        y_combined = data_combined['price'].astype(float)
        
        X_real = data_real[['year', 'mileage', 'make']].astype(float)
        y_real = data_real['price'].astype(float)
        
        X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(
            X_combined, y_combined, test_size=0.2, random_state=42)
        X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
            X_real, y_real, test_size=0.2, random_state=42)
        
        X_train_combined = torch.tensor(X_train_combined.values, dtype=torch.float32)
        y_train_combined = torch.tensor(y_train_combined.values, dtype=torch.float32).view(-1, 1)
        X_test_combined = torch.tensor(X_test_combined.values, dtype=torch.float32)
        y_test_combined = torch.tensor(y_test_combined.values, dtype=torch.float32).view(-1, 1)
        
        X_train_real = torch.tensor(X_train_real.values, dtype=torch.float32)
        y_train_real = torch.tensor(y_train_real.values, dtype=torch.float32).view(-1, 1)
        X_test_real = torch.tensor(X_test_real.values, dtype=torch.float32)
        y_test_real = torch.tensor(y_test_real.values, dtype=torch.float32).view(-1, 1)
        
        train_preds_combined = train_base_models(X_train_combined, y_train_combined.numpy().flatten())
        test_preds_combined = train_base_models(X_test_combined, y_test_combined.numpy().flatten())
        train_preds_real = train_base_models(X_train_real, y_train_real.numpy().flatten())
        test_preds_real = train_base_models(X_test_real, y_test_real.numpy().flatten())
        
        train_data_combined = torch.tensor(train_preds_combined, dtype=torch.float32)
        test_data_combined = torch.tensor(test_preds_combined, dtype=torch.float32)
        train_data_real = torch.tensor(train_preds_real, dtype=torch.float32)
        test_data_real = torch.tensor(test_preds_real, dtype=torch.float32)
        
        meta_model_combined = MetaModel()
        mse_combined, rmse_combined, mae_combined = train_meta_model(
            meta_model_combined, train_data_combined, y_train_combined, test_data_combined, y_test_combined)
        
        meta_model_real = MetaModel()
        mse_real, rmse_real, mae_real = train_meta_model(
            meta_model_real, train_data_real, y_train_real, test_data_real, y_test_real)
        
        owner_results[owner] = {
            'synthetic': {'MSE': mse_combined, 'RMSE': rmse_combined, 'MAE': mae_combined},
            'real': {'MSE': mse_real, 'RMSE': rmse_real, 'MAE': mae_real}
        }
        
        print(f"Owner {owner} - smote data: MSE={mse_combined:.4f}, RMSE={rmse_combined:.4f}, MAE={mae_combined:.4f}")
        print(f"Owner {owner} - real data: MSE={mse_real:.4f}, RMSE={rmse_real:.4f}, MAE={mae_real:.4f}")

    return owner_results

owner_results = evaluate_by_owner(owner_groups_combined, owner_groups_real)

# plot by owner results
owners = list(owner_results.keys())
rmse_real = [owner_results[owner]['real']['RMSE'] for owner in owners]
rmse_synthetic = [owner_results[owner]['synthetic']['RMSE'] for owner in owners]

x = np.arange(len(owners))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, rmse_real, width, label='Real Data RMSE')
plt.bar(x + width/2, rmse_synthetic, width, label='Synthetic Data RMSE')
plt.xlabel('Owner Count')
plt.ylabel('RMSE')
plt.title('RMSE Comparison for Real and Synthetic Data by Owner Count')
plt.xticks(x, owners)
plt.legend()
plt.tight_layout()
plt.savefig('by_owner_results.png',dpi=300)

# plot synthetic data density
plt.figure(figsize=(10, 5))
sns.kdeplot(df_q5['price'], label='Real Data', shade=True)
sns.kdeplot(generated_data['price'], label='Synthetic Data', shade=True)
plt.title(f'Distribution of price for Real and Synthetic Data')
plt.xlabel('price')
plt.ylabel('Density')
plt.legend()
plt.savefig('generate_price_dist.png',dpi=300)

# plot the rmse by brand and overall 
brands = list(results.keys())
rmse_real = [results[brand]['real']['RMSE'] for brand in brands]
rmse_synthetic = [results[brand]['synthetic']['RMSE'] for brand in brands]
mae_real = [results[brand]['real']['MAE'] for brand in brands]
mae_synthetic = [results[brand]['synthetic']['MAE'] for brand in brands]

x = np.arange(len(brands))  
width = 0.2

plt.figure(figsize=(14, 8))

plt.bar(x - width, rmse_real, width, label='Real Data RMSE')
plt.bar(x, rmse_synthetic, width, label='Synthetic Data RMSE')
plt.bar(x + width, mae_real, width, label='Real Data MAE', color='lightblue')
plt.bar(x + 2 * width, mae_synthetic, width, label='Synthetic Data MAE', color='orange')

plt.xlabel('Brand')
plt.ylabel('Metric Value')
plt.title('Comparison of RMSE and MAE by Brand for Real and Synthetic Data')
plt.xticks(x, brands, rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('by_brand_rmse.png',dpi = 300)




