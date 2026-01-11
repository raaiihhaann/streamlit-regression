import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

df = pd.read_csv("train_digicamp.csv")

FEATURES = [
    "GrLivArea",
    "LotArea",
    "TotalBsmtSF",
    "BedroomAbvGr",
    "FullBath",
    "TotRmsAbvGrd",
    "OverallQual",
    "OverallCond",
    "KitchenQual",
    "GarageCars",
    "GarageArea",
    "Neighborhood"
]

TARGET = "SalePrice"

df = df[FEATURES + [TARGET]]

kitchen_mapping = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
df["KitchenQual"] = df["KitchenQual"].map(kitchen_mapping)

neighborhood_mapping = {
    "Blmngtn": 0, "Blueste": 1, "BrDale": 2, "BrkSide": 3,
    "ClearCr": 4, "CollgCr": 5, "Crawfor": 6, "Edwards": 7,
    "Gilbert": 8, "IDOTRR": 9, "MeadowV": 10, "Mitchel": 11,
    "NAmes": 12, "NPkVill": 13, "NWAmes": 14, "NoRidge": 15,
    "NridgHt": 16, "OldTown": 17, "SWISU": 18, "Sawyer": 19,
    "SawyerW": 20, "Somerst": 21, "StoneBr": 22, "Timber": 23,
    "Veenker": 24
}
df["Neighborhood"] = df["Neighborhood"].map(neighborhood_mapping)

df.fillna(0, inplace=True)

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:,.2f}")
print(f"R2  : {r2:.4f}")

joblib.dump(model, "house_price_model.pkl")
joblib.dump(FEATURES, "feature_columns.pkl")