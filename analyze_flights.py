import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# -----------------------
# 📥 LOAD
# -----------------------
df = pd.read_csv("flights_history.csv")
print("Rows loaded:", len(df))

# -----------------------
# 🕒 TIME
# -----------------------
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.sort_values(["icao24", "timestamp"])

# -----------------------
# 📊 FEATURE ENGINEERING
# -----------------------
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek

df["traffic_density"] = df.groupby("timestamp")["icao24"].transform("count")

# 🧠 advanced features
df["is_night"] = ((df["hour"] < 6) | (df["hour"] > 20)).astype(int)

df["velocity_change"] = df.groupby("icao24")["velocity"].diff().fillna(0)

df["high_traffic"] = (
    df["traffic_density"] > df["traffic_density"].median()
).astype(int)

# -----------------------
# CLEAN
# -----------------------
df = df.fillna({
    "velocity": 0,
    "geo_altitude": 0
})

# -----------------------
# 🎯 BETTER DELAY LOGIC
# -----------------------
df["slow_flight"] = (
    (df["velocity"] < 200) &
    (df["geo_altitude"] > 1000)
).astype(int)

df["delay"] = (
    df["slow_flight"] |
    (df["high_traffic"] == 1)
).astype(int)

print("Delay distribution:")
print(df["delay"].value_counts())

# -----------------------
# 🤖 ML MODEL
# -----------------------
features = [
    "hour",
    "day_of_week",
    "traffic_density",
    "velocity",
    "geo_altitude",
    "is_night",
    "velocity_change",
    "high_traffic"
]

df_model = df.dropna(subset=features + ["delay"])

X = df_model[features]
y = df_model["delay"]

if len(df_model) < 100:
    print("Not enough data for ML yet")
    df["delay_prob"] = 0
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print("Model accuracy:", accuracy)

    df["delay_prob"] = model.predict_proba(df[features])[:, 1]

    # 🔥 feature importance
    importances = model.feature_importances_
    print("\nFeature importance:")
    print(list(zip(features, importances)))

# -----------------------
# 🚦 RISK LEVEL
# -----------------------
df["risk_level"] = pd.cut(
    df["delay_prob"],
    bins=[0, 0.3, 0.7, 1],
    labels=["Low", "Medium", "High"]
)

# -----------------------
# 📊 INSIGHTS
# -----------------------
print("\nTop risky hours:")
print(df.groupby("hour")["delay"].mean().sort_values(ascending=False).head())

print("\nTraffic impact:")
print(df.groupby("high_traffic")["delay"].mean())

print("\nNight vs day:")
print(df.groupby("is_night")["delay"].mean())

# -----------------------
# 💾 EXPORT
# -----------------------
df.to_csv("flights_processed.csv", index=False)

print("Saved processed file:", len(df))
