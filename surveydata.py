import csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import cm

pd.set_option("display.max_rows", 21)
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 1000)

# Matplotlib formatting
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 12}

matplotlib.rc('font', **font)

params = {
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
matplotlib.rcParams.update(params)

#TODO: Subscript in axis and plot titles


def create_blank_spreadsheet(path=None, spectra="octaves", lowest_band="63 Hz Leq", highest_band="8000 Hz Leq"):
    assert path is not None

    octave_bands = ["31.5 Hz Leq", "63 Hz Leq", "125 Hz Leq", "250 Hz Leq", "500 Hz Leq", "1000 Hz Leq",
                    "2000 Hz Leq", "4000 Hz Leq", "8000 Hz Leq"]
    third_octave_bands = ["25 Hz Leq", "31.5 Hz Leq", "40 Hz Leq", "50 Hz Leq", "63 Hz Leq", "80 Hz Leq", "Leq 100 Hz Leq",
                          "125 Hz Leq", "160 Hz Leq", "200 Hz Leq", "250 Hz Leq", "315 Hz Leq", "400 Hz Leq",
                          "500 Hz Leq", "630 Hz Leq", "800 Hz Leq", "1000 Hz Leq", "1250 Hz Leq", "1600 Hz Leq",
                          "2000 Hz Leq", "2500 Hz Leq", "3150 Hz Leq", "4000 Hz Leq", "5000 Hz Leq", "6300 Hz Leq",
                          "8000 Hz Leq"]

    centre_freqs = []
    if spectra == "octaves":
        centre_freqs = octave_bands
    elif spectra == "thirds":
        centre_freqs = third_octave_bands
    # Eliminate the bands before the lowest band
    print(centre_freqs)
    lower_bound = centre_freqs.index(lowest_band)
    upper_bound = centre_freqs.index(highest_band)
    centre_freqs = centre_freqs[lower_bound:upper_bound + 1]

    print(centre_freqs)
    table_header = ["Timestamp", "LAeq", "LAFmax", "LA90"]
    print(table_header)
    for band in centre_freqs:
        table_header.append(band)
    print(table_header)
    #TODO: Fix this so that quote marks are removed from centre bands in the csv
    with open(path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=" ")
        writer.writerow(table_header)


class SurveyData:
    def __init__(self, path, drop_start=True, drop_end=True, meas_interval=1, night_start=23, day_start=7,
                 evening_start=None, decimals=1, thirds_to_octaves=False):
        self.decimals = decimals
        self.csv_path = path
        self.master = pd.read_csv(path)

        self.meas_interval = meas_interval

        # Drop first and last row, if required
        if drop_start:
            self.master.drop(index=0, inplace=True)
        if drop_end:
            self.master.drop(index=(len(self.master) - 1), inplace=True)

        self._prep_indices()    # Assign datetime indices
        self.antilogs = self._prep_antilogs()   # Create a copy dataframe of antilogs
        if thirds_to_octaves:
            self._convert_thirds_to_octaves()

        # Assign day, evening, night periods
        self.night_start = night_start
        self.day_start = day_start
        self.evening_start = evening_start

        # Create dataframes and slices for day, evening, nights
        self.nights = self._prep_nights()   # This is a separate dataframe
        self.days = self._prep_days()   # This is a slice of master
        self.evenings = None
        if self.evening_start != self.night_start:
            self.evenings = self._prep_evenings()   # This is a slice of master

    def _prep_indices(self, drop_meter_ts=True):
        self.master["Time"] = pd.DatetimeIndex(self.master["Time"], dayfirst=True)
        self.master.set_index("Time", drop=True, inplace=True)
        if drop_meter_ts:
            self.master.drop(labels="Meter Timestamp", axis=1, inplace=True)
        # Drop NaT indices
        self.master = self.master.loc[pd.notnull(self.master.index)]

    def _prep_antilogs(self):
        return self.master.copy().apply(lambda x: np.power(10, (x/10)))

#https://pandas.pydata.org/docs/user_guide/timeseries.html#indexing
    def _prep_nights(self):
        nights = self.antilogs[(self.antilogs.index.hour >= self.night_start) |
                               (self.antilogs.index.hour < self.day_start)].copy()
        nights.loc[:, "night_dates"] = nights.index
        nights.loc[:, "night_dates"] = nights["night_dates"].apply(
            lambda x: (x - pd.tseries.offsets.DateOffset(days=1)) if x.hour < self.day_start else x)
        nights.set_index(pd.DatetimeIndex(nights["night_dates"]), drop=True, inplace=True)
        nights.drop("night_dates", axis=1, inplace=True)
        return nights

    def _prep_days(self):
        return self.antilogs[(self.antilogs.index.hour >= self.day_start) |
                             (self.antilogs.index.hour < self.evening_start)]

    def _prep_evenings(self):
        return self.antilogs[(self.antilogs.index.hour >= self.evening_start) |
                             (self.antilogs.index.hour < self.night_start)]

    def _daily_leqs(self):
        night_leq = self.nights.groupby(self.nights.index.date).mean().apply(
            lambda x: np.round((10 * np.log10(x)), self.decimals))
        night_leq.drop("LAFmax", axis=1, inplace=True)  # Drop LAFmax column
        day_leq = self.days.groupby(self.days.index.date).mean().apply(
            lambda x: np.round((10 * np.log10(x)), self.decimals))
        day_leq.drop("LAFmax", axis=1, inplace=True)  # Drop LAFmax column
        if self.evening_start is not None:
            evening_leq = self.days.groupby(self.evenings.index.date).mean().apply(
                lambda x: np.round((10 * np.log10(x)), self.decimals))
            evening_leq.drop("LAFmax", axis=1, inplace=True)  # Drop LAFmax column
            return day_leq, evening_leq, night_leq
        else:
            return day_leq, night_leq

    def _convert_thirds_to_octaves(self):
        # Initialise new dataframe
        new_df = pd.DataFrame(self.antilogs[["LAeq", "LAFmax", "LA90"]])

        # Convert Leqs
        leq_cols = self.antilogs.loc[:, self.antilogs.columns.str.contains("Leq", case=True)]
        leq_cols_list = leq_cols.columns.tolist()
        for idx in range(len(leq_cols_list)):
            if (idx + 3) % 3 == 1:
                col_name = leq_cols_list[idx]
                col_before = leq_cols_list[idx - 1]
                col_after = leq_cols_list[idx + 1]
                new_df[col_name] = self.antilogs[[col_before, col_name, col_after]].sum(axis=1)

        # Convert Lmaxes
        lmax_cols = self.antilogs.loc[:, self.antilogs.columns.str.contains("Lmax", case=True)]
        lmax_cols_list = lmax_cols.columns.tolist()
        for idx in range(len(lmax_cols_list)):
            if (idx + 3) % 3 == 1:
                col_name = lmax_cols_list[idx]
                col_before = lmax_cols_list[idx - 1]
                col_after = lmax_cols_list[idx + 1]
                new_df[col_name] = self.antilogs[[col_before, col_name, col_after]].sum(axis=1)

        # TODO: Implement for Ln spectra
        # Assign the single-octave band spectra
        self.antilogs = new_df.copy()

        # Compare dataframes - testing only
        # comp = self.antilogs[["LAeq", "LAFmax", "LA90"]].compare(new_df[["LAeq", "LAFmax", "LA90"]], align_axis=1, keep_shape=True, keep_equal=False)
        # print(comp.isna().sum())

    def _recompute(self, t=10):
        new_df = pd.DataFrame(columns=self.antilogs.columns)
        resampling_string = str(t) + "min"
        leqs = self.antilogs.resample(resampling_string).mean().apply(lambda x: np.round((10 * np.log10(x)), self.decimals))
        leqs = leqs.loc[:, ~leqs.columns.str.contains("Lmax", case=True)]  # Drop Lmaxes
        leqs = leqs.loc[:, ~leqs.columns.str.contains("LAmax", case=True)]  # Drop LAmaxes
        # TODO: Implement LA90s - currently it is taken from the Leq df
        lmaxes = self.antilogs.resample(resampling_string).max().apply(lambda x: np.round((10 * np.log10(x)), self.decimals))
        lmaxes = lmaxes.loc[:, ~lmaxes.columns.str.contains("Leq", case=True)]  # Drop Leqs
        lmaxes = lmaxes.loc[:, ~lmaxes.columns.str.contains("LAeq", case=True)]  # Drop LAeqs
        lmaxes = lmaxes.loc[:, ~lmaxes.columns.str.contains("LA90", case=True)]  # Drop LA90

        # Assign leqs and lmaxes to new df
        for col in leqs.columns:
            new_df[col] = leqs[col]

        for col in lmaxes.columns:
            new_df[col] = lmaxes[col]

        return new_df

    def get_leq_summary(self, drop_cols=["LA90", "31.5 Hz Leq"]):
        if self.evening_start is not None:
            # Drop the lmaxes
            day, evening, night = self._daily_leqs().loc[:, ~self._daily_leqs.columns.str.contains("Lmax", case=True)]
            day.drop(drop_cols, axis=1, inplace=True)
            evening.drop(drop_cols, axis=1, inplace=True)
            night.drop(drop_cols, axis=1, inplace=True)
            return day, evening, night

        else:
            # Drop the lmaxes
            day, night = self._daily_leqs()
            day.drop(drop_cols, axis=1, inplace=True)
            night.drop(drop_cols, axis=1, inplace=True)
            return day.loc[:, ~day.columns.str.contains("Lmax", case=True)],\
                   night.loc[:, ~night.columns.str.contains("Lmax", case=True)]

    def get_nth_lmaxes(self, nth=10, night_only=True, sampling_period=2, drop_cols=["LAeq", "LA90", "31.5 Hz Lmax"]):
        resampling_string = str(sampling_period) + "min"
        if not night_only:
            return None    # TODO: Lmaxes for other periods
        else:
            maxes = self.nights.resample(resampling_string).max().apply(lambda x: np.round((10 * np.log10(x)), self.decimals))
            maxes = maxes[(maxes.index.hour < self.day_start) |
                          (maxes.index.hour >= self.night_start)]   # Resample adds in missing rows. Drop them.
            nth_highest = maxes.sort_values(by="LAFmax", ascending=False)
            nth_highest = nth_highest.groupby(by=nth_highest.index.date).nth(nth - 1)
            nth_highest = nth_highest.loc[:, ~maxes.columns.str.contains("Leq", case=True)] # Drop spectral Leqs
            # Drop unwanted cols
            nth_highest.drop(drop_cols, axis=1, inplace=True)
            return nth_highest

    def lmax_histogram_by_day(self, night_only=True, sampling_period=2):
        resampling_string = str(sampling_period) + "min"
        if not night_only:
            return None # TODO: Lmax histograms for other periods
        else:
            maxes = self.nights.resample(resampling_string).max().apply(lambda x: np.round((10 * np.log10(x)), 0))
            maxes = maxes[(maxes.index.hour < self.day_start) |
                          (maxes.index.hour >= self.night_start)]   # Resample adds in missing rows. Drop them.
            sns.set_theme(style="darkgrid")
            maxes["Day of the week"] = maxes.index.day_name()
            # sns.set(rc={"figure.figsize": (10, 10)})
            ax = sns.catplot(x="Day of the week", y="LAFmax", data=maxes, jitter=True)
            plt.title("LAmax occurrences by day, night-time", pad=-10)
            plt.show()

    def lmax_histogram_count(self, night_only=True, sampling_period=2, stat="count"):
        resampling_string = str(sampling_period) + "min"
        if not night_only:
            return None # TODO: Lmax histograms for other periods
        else:
            maxes = self.nights.resample(resampling_string).max().apply(lambda x: np.round((10 * np.log10(x)), 0))
            maxes = maxes[(maxes.index.hour < self.day_start) |
                          (maxes.index.hour >= self.night_start)]   # Resample adds in missing rows. Drop them.
            sns.set_theme(style="darkgrid")
            maxes["Day of the week"] = maxes.index.day_name()
            ax = sns.histplot(x="LAFmax", data=maxes, binwidth=1, stat=stat)
            plt.title("LAmax occurrences, night-time")
            plt.show()

    def time_history_plot(self, sampling_period=15):
        # Define the colourmap
        viridis = cm.get_cmap("plasma")
        col = iter(np.linspace(0.0, 1, 4))

        # Resample to a lower frequency for speedier computation and better presentation
        prepped_df = self._recompute(t=sampling_period)
        print("plotting....")
        sns.set_style("whitegrid")
        # pal = iter(sns.color_palette("flare", 6))
        # TODO: Sort out sizes, fonts etc.
        fig, ax1 = plt.subplots(figsize=(14, 10))
        ax1.set_facecolor("xkcd:pale grey")
        ax1.set_ylabel("Sound pressure level dB(A)")
        ax1.set_xlabel("Date")
        # TODO: implement custom colours
        ax1.plot(prepped_df.index, prepped_df["LAeq"], color=viridis(next(col)), label="LAeq,T", lw=3)
        ax1.plot(prepped_df.index, prepped_df["LA90"], color=viridis(next(col)), label="LA90,T", lw=3)
        ax1.scatter(prepped_df.index, prepped_df["LAFmax"], s=8, color=viridis(next(col)), label="LAFmax", linewidths=3)
        ax1.legend()
        # TODO: Try this with shorter or longer surveys
        # TODO: show hours at relevant intervals
        # Configure major ticks at midnight
        major_locator = matplotlib.ticker.FixedLocator(np.unique(prepped_df.index.date))
        ax1.xaxis.set_major_formatter(
            mdates.DateFormatter("%d %b\n%r"))
        # Configure minor ticks every 6 hours
        # ax1.minorticks_on()
        # hour_locator = matplotlib.dates.HourLocator(interval=6)
        # ax1.xaxis.set_minor_locator(hour_locator)
        # Grid
        for axis in ['top', 'bottom', 'left', 'right']:
            ax1.spines[axis].set_linewidth(1.5)
            ax1.spines[axis].set_color("black")
        ax1.grid(True, which="major", lw=1.5, color="black")   # Thicker lines on major ticks
        ax1.grid(True, which="minor", axis="x", lw=0.5, color="black")     # Thinner lines on minor ticks
        print("showing....")
        plt.title("Time history")
        plt.show()
        print("done")

    def l90_histogram_count(self, night_only=False, night_t=15, day_t=60, stat="count"):

        if not night_only:
            day_resampling_string = str(day_t) + "min"
            l90 = self.antilogs.resample(day_resampling_string).mean().apply(lambda x: np.round((10 * np.log10(x)), 0))
            l90 = l90[(l90.index.hour >= self.day_start) |
                          (l90.index.hour < self.night_start)]   # Resample adds in missing rows. Drop them.
            sns.set_theme(style="darkgrid")
            l90["Day of the week"] = l90.index.day_name()
            ax = sns.histplot(x="LA90", data=l90, binwidth=1, stat=stat)
            plt.title("LA90,%s, occurrences, daytime" % day_resampling_string)
            plt.show()

        night_resampling_string = str(night_t) + "min"
        l90 = self.nights.resample(night_resampling_string).mean().apply(lambda x: np.round((10 * np.log10(x)), 0))
        l90 = l90[(l90.index.hour < self.day_start) |
                      (l90.index.hour >= self.night_start)]   # Resample adds in missing rows. Drop them.
        sns.set_theme(style="darkgrid")
        l90["Day of the week"] = l90.index.day_name()
        ax = sns.histplot(x="LA90", data=l90, binwidth=1, stat=stat)
        plt.title("LA90,%s occurrences, night-time" % night_resampling_string)
        plt.show()

