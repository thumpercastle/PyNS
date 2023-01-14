import requests
import pandas as pd
import datetime as dt


appid = ""
with open("openweather_appid.txt") as f:
    appid = f.readlines()[0]

w_dict = {
    "start": "2022-09-16 12:00:00",
    "end": "2022-09-17 18:00:00",
    "interval": 24,
    "api_key": appid,
    "lat": 51.4780,
    "lon": 0.0015,
    "tz": "GB"
}

def test_weather_obj(weather_test_dict):
    hist = WeatherHistory(start=w_dict["start"], end=w_dict["end"], interval=w_dict["interval"],
                          api_key=w_dict["api_key"], lat=w_dict["lat"], lon=w_dict["lon"], tz=w_dict["tz"])
    hist.compute_weather_history()
    return hist


class WeatherHistory:
    def __init__(self, start="", end="", interval=6, api_key="", lat=00.00001,
                 lon=00.0001, tz=""):
        self._history_df = pd.DataFrame()
        self._start = dt.datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
        self._end = dt.datetime.strptime(end, "%Y-%m-%d %H:%M:%S")
        self._interval = interval
        self._api_key = str(api_key)
        self._lat = str(lat)
        self._lon = str(lon)
        self._hist = None

    def _construct_api_call(self, timestamp):
        base = "https://api.openweathermap.org/data/3.0/onecall/timemachine?"
        query = str(base + "lat=" + self._lat + "&" + "lon=" + self._lon + "&" + "dt=" + str(timestamp) + "&" + "appid=" + \
              self._api_key)
        print(query)
        return query

    def _construct_timestamps(self):
        next_time = (self._start + dt.timedelta(hours=self._interval))
        timestamps = [int(self._start.timestamp())]
        while next_time < self._end:
            timestamps.append(int(next_time.timestamp()))
            next_time += dt.timedelta(hours=self._interval)
        return timestamps

    def _make_and_parse_api_call(self, query):
        response = requests.get(query)
        print(response.json())
        # This drops some unwanted cols like lat, lon, timezone and tz offset.
        resp_dict = response.json()["data"][0]
        del resp_dict["weather"]    # delete weather key as not useful.
        # TODO: parse 'weather' nested dict.
        return resp_dict

    def compute_weather_history(self):
        # construct timestamps
        timestamps = self._construct_timestamps()
        # make calls to API
        responses = []
        for ts in timestamps:
            print(f"ts: {ts}")
            query = self._construct_api_call(timestamp=ts)
            response_dict = self._make_and_parse_api_call(query=query)
            responses.append(pd.Series(response_dict))
        df = pd.concat(responses, axis=1).transpose()
        for col in ["dt", "sunrise", "sunset"]:
            df[col] = df[col].apply(lambda x: dt.datetime.fromtimestamp(int(x)))  # convert timestamp into datetime
        print(df)
        self._hist = df
        return df

    def get_weather_history(self):
        return self._hist
