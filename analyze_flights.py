import pandas as pd
import numpy as np
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

df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek

# -----------------------
# 📊 TRAFFIC
# -----------------------
df["traffic_density"] = df.groupby("timestamp")["icao24"].transform("count")

# 🔥 better traffic threshold (top 25%)
traffic_threshold = df["traffic_density"].quantile(0.75)
df["high_traffic"] = (df["traffic_density"] > traffic_threshold).astype(int)

# -----------------------
# 🧠 TIME BUCKET
# -----------------------
df["time_of_day"] = pd.cut(
    df["hour"],
    bins=[0,6,12,18,24],
    labels=["Night","Morning","Afternoon","Evening"]
)

df["time_of_day_num"] = df["time_of_day"].map({
    "Night": 0,
    "Morning": 1,
    "Afternoon": 2,
    "Evening": 3
})

# -----------------------
# ⚡ VELOCITY FEATURES
# -----------------------
df["velocity_change"] = df.groupby("icao24")["velocity"].diff().fillna(0)

# 🔥 mai stabil decât raw change
df["is_slowing"] = (df["velocity_change"] < -10).astype(int)

# -----------------------
# CLEAN
# -----------------------
df = df.fillna({
    "velocity": 0,
    "geo_altitude": 0,
    "time_of_day_num": 0
})

# -----------------------
# 🎯 DELAY LOGIC
# -----------------------
df["slow_flight"] = (
    (df["velocity"] < 200) &
    (df["geo_altitude"] > 1000)
).astype(int)

df["delay"] = (
    df["slow_flight"] |
    df["high_traffic"]
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
    "is_slowing",
    "time_of_day_num"
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

    X_full = df[features].fillna(0)
    df["delay_prob"] = model.predict_proba(X_full)[:, 1]

    # 🔥 feature importance
    importances = model.feature_importances_
    print("\nFeature importance:")
    print(dict(zip(features, importances)))

# -----------------------
# 🚦 CLEAN PROB
# -----------------------
df["delay_prob"] = df["delay_prob"].fillna(0)

# -----------------------
# 🚦 RISK LEVEL (robust)
# -----------------------
conditions = [
    (df["delay_prob"] <= 0.3),
    (df["delay_prob"] <= 0.7),
    (df["delay_prob"] > 0.7)
]

choices = ["Low", "Medium", "High"]

df["risk_level"] = np.select(conditions, choices, default="Low")

# -----------------------
# 🧠 EXPLAINABILITY
# -----------------------
def explain_row(row):
    reasons = []

    if row["high_traffic"] == 1:
        reasons.append("High traffic")

    if row["velocity"] < 200:
        reasons.append("Low speed")

    if row["is_slowing"] == 1:
        reasons.append("Slowing down")

    if row["geo_altitude"] < 2000:
        reasons.append("Low altitude")

    return ", ".join(reasons)

df["delay_reason"] = df.apply(
    lambda row: explain_row(row) if row["delay_prob"] > 0.6 else "No risk",
    axis=1
)

# -----------------------
# 📊 INSIGHTS
# -----------------------
print("\nTop risky hours:")
print(df.groupby("hour")["delay"].mean().sort_values(ascending=False).head())

print("\nTraffic impact:")
print(df.groupby("high_traffic")["delay"].mean())

print("\nTime of day:")
print(df.groupby("time_of_day")["delay"].mean())

# -----------------------
# 💾 EXPORT
# -----------------------
df.to_csv("flights_processed.csv", index=False)

print("Saved processed file:", len(df))
