import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load datasets
orders = pd.read_csv("data/orders.csv")
delivery = pd.read_csv("data/delivery_performance.csv")
routes = pd.read_csv("data/routes_distance.csv")
customers = pd.read_csv("data/customer_feedback.csv")
costs = pd.read_csv("data/cost_breakdown.csv")

# -----------------------------
# Create target variable
# -----------------------------
delivery["delay_days"] = (
    delivery["Actual_Delivery_Days"] - delivery["Promised_Delivery_Days"]
)

delivery["delayed"] = delivery["delay_days"].apply(
    lambda x: 1 if x > 0 else 0
)

# -----------------------------
# Merge datasets using Order_ID
# -----------------------------
df = orders.merge(delivery, on="Order_ID", how="inner")
df = df.merge(routes, on="Order_ID", how="left")
df = df.merge(customers[["Order_ID", "Rating", "Would_Recommend"]], on="Order_ID", how="left")
df = df.merge(costs, on="Order_ID", how="left")

print("Final columns:")
print(df.columns)
print("\nShape:", df.shape)
print(df.head())


# -----------------------------
# Feature selection
# -----------------------------
features = df[
    [
        "Priority",
        "Product_Category",
        "Order_Value_INR",
        "Distance_KM",
        "Traffic_Delay_Minutes",
        "Fuel_Cost",
        "Labor_Cost",
        "Delivery_Cost_INR",
        "Rating"
    ]
]

target = df["delayed"]


# Fill missing values
features = features.fillna(features.median(numeric_only=True))
features = features.fillna("Unknown")


# Save encoders
encoders = {}

for col in features.columns:
    if features[col].dtype == "object":
        le = LabelEncoder()
        features[col] = le.fit_transform(features[col])
        encoders[col] = le

# Scale numeric features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

#training
X_train, X_test, y_train, y_test = train_test_split(
    features_scaled, target, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


with open("delay_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
