import pandas as pd

# load data
df = pd.read_csv("flights_history.csv")

# convert time
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# sort
df = df.sort_values(["icao24", "timestamp"])

# feature engineering
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek

df["traffic_density"] = df.groupby("timestamp")["icao24"].transform("count")

# delay logic (simplu)
df["delay"] = (
    (df["velocity"] < 200) & 
    (df["geo_altitude"] > 1000)
).astype(int)

print(df.head())
