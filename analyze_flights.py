import pandas as pd

df = pd.read_csv("flights_history.csv")

print("Rows loaded:", len(df))

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

df = df.sort_values(["icao24", "timestamp"])

df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek

df["traffic_density"] = df.groupby("timestamp")["icao24"].transform("count")

df = df.fillna({
    "velocity": 0,
    "geo_altitude": 0
})

df["delay"] = (
    (df["velocity"] < 200) & 
    (df["geo_altitude"] > 1000)
).astype(int)

print(df.head())

df.to_csv("flights_processed.csv", index=False)

print("Saved:", len(df))
