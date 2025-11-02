import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

data = pd.read_csv("data/train.csv")

# fjerne unødvendig kolonner
data = data.drop(columns=["id", "ext_col", "int_col"])

# --- Step 3: Fill missing values ---
data["fuel_type"] = data["fuel_type"].fillna("Ukjent")
data["accident"] = data["accident"].fillna("Ukjent")
data["clean_title"] = data["clean_title"].fillna("Ukjent")
data["model"] = data["model"].fillna("Ukjent")
data["engine"] = data["engine"].fillna("Ukjent")

# fjern de som er for høye for å gjøre mer nøyaktig
data = data[data["price"] < 1_000_000]


for col in ["model", "engine"]:
    avg_price = data.groupby(col)["price"].mean()
    data[col + "_encoded"] = data[col].map(avg_price)
data = data.drop(columns=["model", "engine"])

# One-hot encoding for remaining categorical columns
categorical_cols = ["brand", "fuel_type", "transmission", "accident", "clean_title"]
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)


X = data.drop(columns=["price"])
y = data["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training set:", X_train.shape)
print("Testing set:", X_test.shape)

# trene modell
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# få statistikk
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R2:", r2)

#lagre modell
joblib.dump(model, "trained_car_price_model.pkl")
print("Model saved as trained_car_price_model.pkl")
