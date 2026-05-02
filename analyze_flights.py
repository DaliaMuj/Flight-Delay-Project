import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# -----------------------
# 📥 LOAD
# -----------------------
def load_data(path="flights_history.csv"):
    df = pd.read_csv(path)
    print("Rows loaded:", len(df))
    return df


# -----------------------
# 🕒 TIME FEATURES
# -----------------------
def add_time_features(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values(["icao24", "timestamp"])

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    return df


# -----------------------
# 📊 TRAFFIC FEATURES
# -----------------------
def add_traffic_features(df):
    df["traffic_density"] = df.groupby("timestamp")["icao24"].transform("count")

    threshold = df["traffic_density"].quantile(0.75)
    df["high_traffic"] = (df["traffic_density"] > threshold).astype(int)

    return df


# -----------------------
# 🧠 TIME OF DAY
# -----------------------
def add_time_of_day(df):
    df["time_of_day"] = pd.cut(
        df["hour"],
        bins=[0, 6, 12, 18, 24],
        labels=["Night", "Morning", "Afternoon", "Evening"]
    )

    df["time_of_day_num"] = df["time_of_day"].map({
        "Night": 0,
        "Morning": 1,
        "Afternoon": 2,
        "Evening": 3
    })

    return df


# -----------------------
# ⚡ SPEED + ALTITUDE FEATURES
# -----------------------
def add_physics_features(df):
    df["speed_level"] = pd.cut(
        df["velocity"],
        bins=[0, 150, 250, 400],
        labels=["Low", "Medium", "High"]
    )

    df["speed_level_num"] = df["speed_level"].map({
        "Low": 0,
        "Medium": 1,
        "High": 2
    })

    df["altitude_level"] = pd.cut(
        df["geo_altitude"],
        bins=[0, 2000, 8000, 12000],
        labels=["Low", "Cruise", "High"]
    )

    df["altitude_level_num"] = df["altitude_level"].map({
        "Low": 0,
        "Cruise": 1,
        "High": 2
    })

    return df


# -----------------------
# CLEAN
# -----------------------
def clean_data(df):
    df = df.fillna({
        "velocity": 0,
        "geo_altitude": 0,
        "time_of_day_num": 0,
        "speed_level_num": 0,
        "altitude_level_num": 0
    })
    return df


# -----------------------
# 🎯 TARGET (DELAY)
# -----------------------
def create_target(df):
    df["slow_flight"] = (
        (df["velocity"] < 200) &
        (df["geo_altitude"] > 1000)
    ).astype(int)

    df["delay"] = (
        df["slow_flight"] |
        df["high_traffic"]
    ).astype(int)

    print("\nDelay distribution:")
    print(df["delay"].value_counts())

    return df


# -----------------------
# 🤖 ML MODEL
# -----------------------
def train_model(df):
    features = [
        "hour",
        "day_of_week",
        "traffic_density",
        "velocity",
        "geo_altitude",
        "high_traffic",
        "time_of_day_num",
        "speed_level_num",
        "altitude_level_num"
    ]

    df_model = df.dropna(subset=features + ["delay"])

    if len(df_model) < 100:
        print("Not enough data for ML")
        df["delay_prob"] = 0.1
        return df

    X = df_model[features]
    y = df_model["delay"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )

    model = RandomForestClassifier(
            n_estimators=300,
            max_depth=7,
            min_samples_split=15,
            min_samples_leaf=8,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1
    )

    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print("\nModel accuracy:", accuracy)

    X_full = df[features].fillna(0)
    df["delay_prob"] = model.predict_proba(X_full)[:, 1]

    # 🔥 FIX: evităm 0 și 1
    df["delay_prob"] = df["delay_prob"].clip(0.01, 0.99)

    importances = model.feature_importances_
    print("\nFeature importance:")
    print(dict(zip(features, importances)))

    return df


# -----------------------
# 🚦 RISK LEVEL
# -----------------------
def add_risk_level(df):
    df["delay_prob"] = df["delay_prob"].fillna(0)

    df["risk_level"] = np.select(
        [
            df["delay_prob"] <= 0.3,
            (df["delay_prob"] > 0.3) & (df["delay_prob"] <= 0.7),
            df["delay_prob"] > 0.7
        ],
        [
            "Low",
            "Medium",
            "High"
        ],
        default="Low"
    )

    return df


# -----------------------
# 🧠 EXPLAINABILITY
# -----------------------
def explain_delay(row):
    reasons = []

    if row["high_traffic"] == 1:
        reasons.append("High traffic")

    if row["velocity"] < 200:
        reasons.append("Low speed")

    if row["geo_altitude"] < 2000:
        reasons.append("Low altitude")

    return ", ".join(reasons)


def add_explanations(df):
    df["delay_reason"] = df.apply(
        lambda row: explain_delay(row)
        if row["delay_prob"] > 0.6
        else "No risk",
        axis=1
    )
    return df


# -----------------------
# 📊 INSIGHTS
# -----------------------
def print_insights(df):
    print("\nTop risky hours:")
    print(
        df.groupby("hour")["delay"]
        .mean()
        .sort_values(ascending=False)
        .head()
    )

    print("\nTraffic impact:")
    print(df.groupby("high_traffic")["delay"].mean())

    print("\nTime of day:")
    print(df.groupby("time_of_day")["delay"].mean())


# -----------------------
# 💾 EXPORT
# -----------------------
def save_output(df, path="flights_processed.csv"):
    df.to_csv(path, index=False)
    print("\nSaved:", len(df))


# -----------------------
# 🚀 MAIN
# -----------------------
def main():
    df = load_data()

    df = add_time_features(df)
    df = add_traffic_features(df)
    df = add_time_of_day(df)
    df = add_physics_features(df)
    df = clean_data(df)
    df = create_target(df)

    df = train_model(df)
    df = add_risk_level(df)
    df = add_explanations(df)

    print_insights(df)
    save_output(df)


if __name__ == "__main__":
    main()
