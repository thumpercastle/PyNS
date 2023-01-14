from surveydata import *
from export import *
from weatherdata import *


if __name__ == '__main__':
    # Where is our noise survey data?
    path = r"Data.csv"
    # Create NoiseSurvey object (handles the processing of noise data)
    data = SurveyData(path, decimals=1, thirds_to_octaves=False)
    # Create Export object (handles the conversion of noise data into MS Word format)
    report = DataExport()

    # Get N-th highest Lmax levels for each night, resampling the data where required
    print(f"Lmaxes")
    print(data.get_nth_lmaxes(nth=10, sampling_period=2))
    report.add_table(data.get_nth_lmaxes(), "Night-time Lmax") # Add an Lmax table to the MS Word file

    # Get Daytime and Night-time Leqs and add tables to the MS Word file
    print(f"Leqs")
    print(data.get_leq_summary())
    report.add_table(data.get_leq_summary()[0], "Daytime Leq")
    report.export(path=r"C:\Users\tonyr\OneDrive - Timbral\Projects\2122215_7c_Northgate_End\data")
    report.add_table(data.get_leq_summary()[1], "Night-time Leq")
    report.export(path=r"C:\Users\tonyr\OneDrive - Timbral\Projects\2122215_7c_Northgate_End\data")

    # Plots
    data.lmax_histogram_by_day()
    data.lmax_histogram_count()
    data.time_history_plot()
    data.l90_histogram_count()

    # Weather history requires you to sign up at openweathermap.com for the One Call API
    # Then store your API key in a text file named "openweather_appid.txt"
    # weather_hist = test_weather_obj(w_dict)
    # print(weather_hist)

    # Export the MS Word file
    report.export(path=os.getcwd())
