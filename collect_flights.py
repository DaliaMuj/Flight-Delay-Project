import requests
import pandas as pd
from datetime import datetime
import os

url = "https://opensky-network.org/api/states/all"

response = requests.get(url)
data = response.json()

states = data.get("states", [])

if states:
    df = pd.DataFrame(states, columns=[
        "icao24","callsign","origin_country","time_position",
        "last_contact","longitude","latitude","baro_altitude",
        "on_ground","velocity","heading","vertical_rate",
        "sensors","geo_altitude","squawk","spi","position_source"
    ])

    df["timestamp"] = datetime.utcnow()

    file_path = "flights_history.csv"

    # dacă există deja fișier → îl citim
    if os.path.isfile(file_path):
        old_df = pd.read_csv(file_path)

        old_df["timestamp"] = pd.to_datetime(old_df["timestamp"])
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=2)

        old_df = old_df[old_df["timestamp"] > cutoff]

        df = pd.concat([old_df, df])

    df.to_csv(file_path, index=False)

    print("Saved:", len(df))
else:
    print("No data")
