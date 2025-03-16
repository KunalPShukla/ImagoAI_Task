import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Path to your training dataset
train_data_path = "https://raw.githubusercontent.com/KunalPShukla/ImagoAI_Task/refs/heads/main/MLE-Assignment.csv"

# Load the dataset
df = pd.read_csv(train_data_path)

# Drop the first column (e.g., sample ID or non-numeric values)
df.drop(df.columns[0], axis=1, inplace=True)

# Use only feature columns (exclude the target variable if it's the last column)
X_train = df.iloc[:, :-1].values  # All columns except the last (target)

# Create and fit the StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

# Save the scaler as scaler.pkl
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Scaler successfully created and saved as scaler.pkl!")
