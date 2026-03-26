import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv("dataset.csv")

# Fill missing values
df.fillna("None", inplace=True)

# Split features and target
X = df.drop("Disease", axis=1)
y = df["Disease"]

# One-hot encoding (IMPORTANT FIX)
X = pd.get_dummies(X)

# Encode target labels
le = LabelEncoder()
y = le.fit_transform(y)

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save files
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(X.columns, open("columns.pkl", "wb"))
pickle.dump(le, open("label_encoder.pkl", "wb"))

print("✅ Model trained and saved successfully!")