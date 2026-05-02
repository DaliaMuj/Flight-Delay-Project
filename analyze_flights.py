import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# LOAD
df = pd.read_csv("flights_history.csv")
print("Rows loaded:", len(df))

# TIME
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# SORT
df = df.sort_values(["icao24", "timestamp"])

# FEATURES
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek

df["traffic_density"] = df.groupby("timestamp")["icao24"].transform("count")

# CLEAN
df = df.fillna({
    "velocity": 0,
    "geo_altitude": 0
})

# TARGET (delay simplu)
df["delay"] = (
    (df["velocity"] < 200) & 
    (df["geo_altitude"] > 1000)
).astype(int)

print("Delay distribution:")
print(df["delay"].value_counts())

# -----------------------
# 🤖 ML MODEL
# -----------------------

features = ["hour", "day_of_week", "traffic_density", "velocity", "geo_altitude"]

df_model = df.dropna(subset=features + ["delay"])

X = df_model[features]
y = df_model["delay"]

# dacă ai prea puține date, evită crash
if len(df_model) < 100:
    print("Not enough data for ML yet")
    df["delay_prob"] = 0
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier(n_estimators=50)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print("Model accuracy:", accuracy)

    # probabilitate delay
    df["delay_prob"] = model.predict_proba(df[features])[:, 1]

# -----------------------
# 📊 INSIGHTS
# -----------------------

print("\nDelay by hour:")
print(df.groupby("hour")["delay"].mean())

print("\nDelay by traffic (top):")
print(df.groupby("traffic_density")["delay"].mean().head())

# -----------------------
# 💾 EXPORT
# -----------------------

df.to_csv("flights_processed.csv", index=False)
print("Saved processed file:", len(df))
