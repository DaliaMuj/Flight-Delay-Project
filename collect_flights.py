import requests
import pandas as pd
from datetime import datetime
import os

url = "https://opensky-network.org/api/states/all"

try:
    response = requests.get(url, timeout=10)

    if response.status_code != 200:
        print("API error:", response.status_code)
        exit()

    data = response.json()

except Exception as e:
    print("Request failed:", e)
    exit()

states = data.get("states") or []

if len(states) == 0:
    print("No data received")
    exit()

df = pd.DataFrame(states, columns=[
    "icao24","callsign","origin_country","time_position",
    "last_contact","longitude","latitude","baro_altitude",
    "on_ground","velocity","heading","vertical_rate",
    "sensors","geo_altitude","squawk","spi","position_source"
])

df["timestamp"] = datetime.utcnow()

file_path = "flights_history.csv"

# citire safe
if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
    try:
        old_df = pd.read_csv(file_path)

        old_df["timestamp"] = pd.to_datetime(old_df["timestamp"], errors="coerce")
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=7)

        old_df = old_df[old_df["timestamp"] > cutoff]

        df = pd.concat([old_df, df], ignore_index=True)

    except Exception as e:
        print("CSV read error:", e)

# salvare
df.to_csv(file_path, index=False)

print("Saved rows:", len(df))
