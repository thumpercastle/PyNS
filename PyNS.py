import os
import docx
import pandas as pd
import numpy as np
import datetime as dt
from docx.shared import Cm
from docx.enum.section import WD_SECTION, WD_ORIENTATION

# TODO: Implement thirds to octaves
# TODO: Implement Plotter class
# TODO: Implement WeatherReporter class
# TODO: Implement Reporter class
# TODO: Write tests


pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 20)

DECIMALS = 1

def get_test_subjects():
    log1 = Log("Pos1_Parsed.csv")
    log2 = Log("Pos2_Parsed.csv")
    survey = Survey()
    survey.add_log(log1, "P1")
    survey.add_log(log2, "P2")
    return survey, log1, log2


class Log:
    def __init__(self, path="", datetime_format=None):
        self._filepath = path
        self._master = pd.read_csv(path, index_col="Time", parse_dates=["Time"],
                                   date_parser=lambda col: pd.to_datetime(col, dayfirst=True, errors="coerce",
                                                                          format=datetime_format))
        self._master = self._master.sort_index(axis=1)
        self._start = self._master.index.min()
        self._end = self._master.index.max()
        self._assign_header()

        # Assign day, evening, night periods
        self._night_start = None
        self._day_start = None
        self._evening_start = None
        self.set_periods()

        # Prepare night-time indices and antilogs
        self._antilogs = self._prep_antilogs()
        self._master = self._append_night_idx(data=self._master)
        self._antilogs = self._append_night_idx(data=self._antilogs)

        self._decimals = 1

    def _assign_header(self):
        csv_headers = self._master.columns.to_list()
        superheaders = [item.split(" ")[0] for item in csv_headers]
        subheaders = [item.split(" ")[1] for item in csv_headers]
        # Convert numerical subheaders to ints
        for i in range(len(subheaders)):
            try:
                subheaders[i] = float(subheaders[i])
            except Exception:
                continue
        self._master.columns = [superheaders, subheaders]
        self._master.sort_index(axis=1, level=1, inplace=True)

    def _prep_antilogs(self):
        return self._master.copy().apply(lambda x: np.power(10, (x / 10)))

    def _append_night_idx(self, data=None):
        night_indices = data.index.to_list()
        if self._night_start > self._day_start:
            for i in range(len(night_indices)):
                if night_indices[i].time() < self._day_start:
                    night_indices[i] += dt.timedelta(days=-1)
        data["Night idx"] = night_indices
        return data

    def _return_as_night_idx(self, data=None):
        if ("Night idx", "") not in data.columns:
            raise Exception("No night indices in current DataFrame")
        return data.set_index("Night idx")

    def _none_if_zero(self, df):
        if len(df) == 0:
            return None
        else:
            return df

    def _recompute_leq(self, data=None, t="15min", cols=None):
        # Set default mutable args
        if data is None:
            data = self._antilogs
        if cols is None:
            cols = ["Leq", "L90"]
        # Loop through column superheaders and recompute as a longer Leq
        recomputed = pd.DataFrame(columns=data.columns)
        for idx in cols:
            if idx in data.columns:
                recomputed[idx] = data[idx].resample(t).mean().\
                    apply(lambda x: np.round((10 * np.log10(x)), self._decimals))
        return self._none_if_zero(recomputed)

    def _recompute_night_idx(self, data=None, t="15min"):
        if data is None:
            raise Exception("No DataFrame provided for night idx")
        if ("Night idx", "") in data.columns:
            data["Night idx"] = data["Night idx"].resample(t).asfreq()
        else:
            data["Night idx"] = self._master["Night idx"].resample(t).asfreq()
            return data

    def _recompute_max(self, data=None, t="15min", pivot_cols=None, hold_spectrum=False):
        # Set default mutable args
        if pivot_cols is None:
            pivot_cols = [("Lmax", "A")]
        if data is None:
            data = self._master
        # Loop through column superheaders and recompute over a longer period
        combined = pd.DataFrame(columns=data.columns)
        if hold_spectrum:   # Hold the highest value, in given period per frequency band
            for col in pivot_cols:
                if col in combined.columns:
                    max_hold = data.resample(t)[col[0]].max()
                    combined[col[0]] = max_hold
        else:   # Event spectrum (octave band data corresponding to the highest A-weighted event)
            for col in pivot_cols:
                if col in combined.columns:
                    idx = data[col[0]].groupby(pd.Grouper(freq=t)).max()
                    combined[col[0]] = idx
        return combined

    def _as_multiindex(self, df=None, super=None, name1="Date", name2="Num"):
        subs = df.index.to_list()   # List of subheaders
        # Super will likely be the date
        tuples = [(super, sub) for sub in subs]
        idx = pd.MultiIndex.from_tuples(tuples, names=[name1, name2])
        if isinstance(df, pd.Series):
            df = pd.DataFrame(data=df)
        return df.set_index(idx, inplace=False)

    # ###########################---PUBLIC---######################################

    def get_antilogs(self):
        return self._antilogs

    def get_period(self, data=None, period="days", night_idx=True):
        if data is None:
            data = self._master
        if period == "days":
            return data.between_time(self._day_start, self._evening_start, inclusive="left")
        elif period == "evenings":
            return data.between_time(self._evening_start, self._night_start, inclusive="left")
        elif period == "nights":
            #print("data before night idx")
            #print(data)
            if night_idx:
                data = self._return_as_night_idx(data=data)
                #print("data after night idx")
                #print(data)
                #print("data between time")
                #print(data.between_time(self._night_start, self._day_start, inclusive="left"))
            return data.between_time(self._night_start, self._day_start, inclusive="left")

    def leq_by_date(self, data, cols=None):
        # This takes antilogs for a given period.
        if cols is None:
            cols = ["Leq"]
        return data[cols].groupby(data.index.date).mean().apply(lambda x: np.round((10 * np.log10(x)), self._decimals))

    def as_interval(self, data=None, antilogs=None, t="15min", leq_cols=None, max_pivots=None,
                    min_pivots=None,
                    hold_spectrum=False):
        # TODO: Implement min
        # Set defaults for mutable args
        if data is None:
            data = self._master
        if antilogs is None:
            antilogs = self._antilogs
        if leq_cols is None:
            leq_cols = ["Leq", "L90"]
        if max_pivots is None:
            max_pivots = [("Lmax", "A")]
        # if min_pivots is None:
        #     min_pivots = []
        leq = self._recompute_leq(data=antilogs, t=t, cols=leq_cols)
        maxes = self._recompute_max(data=data, t=t, pivot_cols=max_pivots, hold_spectrum=hold_spectrum)
        # mins = self._recompute_min(data=data, t=t, pivot_cols=min_pivots, hold_spectrum=hold_spectrum)
        conc = pd.concat([leq, maxes], axis=1).sort_index(axis=1).dropna(axis=1, how="all")
        conc = self._append_night_idx(data=conc)    # Re-append night indices
        return conc.dropna(axis=0, how="all")

    def get_nth_high_low(self, n=10, data=None, pivot_col=None, all_cols=False, high=True):
        if data is None:
            data = self._master
        if pivot_col is None:
            pivot_col = ("Lmax", "A")
        nth = None
        if high:
            nth = data.sort_values(by=pivot_col, ascending=False)
        if not high:
            nth = data.sort_values(by=pivot_col, ascending=True)
        nth["Time"] = nth.index.time
        if all_cols:
            return nth.groupby(by=nth.index.date).nth(n-1)
        else:
            return nth[[pivot_col[0], "Time"]].groupby(by=nth.index.date).nth(n-1)

    def get_modal(self, data=None, by_date=True, cols=None, round_decimals=True):
        if data is None:
            data = self._master
        if round_decimals:
            data = data.round()
        if cols is None:
            cols = [("L90", "A"), ("L90", 8000.0)]
        if by_date:
            dates = np.unique(data.index.date)
            modes_by_date = pd.DataFrame()
            for date in range(len(dates)):
                date_str = dates[date].strftime("%Y-%m-%d")
                mode_by_date = data[cols].loc[date_str].mode()
                mode_by_date = self._as_multiindex(df=mode_by_date, super=date_str)
                modes_by_date = pd.concat([modes_by_date, mode_by_date])
            return modes_by_date
        else:
            #TODO: Implement by_date=False
            pass
            return 1

    def set_periods(self, times=None):
        # Assumes no evening period
        if times is None:
            times = {"day": (7, 0), "evening": (23, 0), "night": (23, 0)}
        self._day_start = dt.time(times["day"][0], times["day"][1])
        self._evening_start = dt.time(times["evening"][0], times["evening"][1])
        self._night_start = dt.time(times["night"][0], times["night"][1])

    def get_period_times(self):
        return self._day_start, self._evening_start, self._night_start

    def is_evening(self):
        if self._evening_start == self._night_start:
            return False
        else:
            return True


class Survey:
    def __init__(self):
        self._logs = {}

    def add_log(self, data=None, name=""):
        self._logs[name] = data

    def _insert_multiindex(self, df=None, super=None, name1="Position", name2="Date"):
        subs = df.index.to_list()   # List of subheaders (dates)
        # Super should be the position name (key from master dictionary)
        tuples = [(super, sub) for sub in subs]
        idx = pd.MultiIndex.from_tuples(tuples, names=[name1, name2])
        return df.set_index(idx, inplace=False)

    def _insert_header(self, df=None, new_head_list=None, header_idx=None):
        cols = df.columns.to_list()
        new_cols = [list(c) for c in zip(*cols)]
        new_cols.insert(header_idx, new_head_list)
        df.columns = new_cols
        return df

    def set_periods(self, times=None):
        if times is None:
            times = {"day": (7, 0), "evening": (23, 0), "night": (23, 0)}
        for key in self._logs.keys():
            self._logs[key].set_periods(times=times)

    def get_periods(self):
        periods = {}
        for key in self._logs.keys():
            periods[key] = self._logs[key].get_period_times()
        return periods

    def resi_summary(self, leq_cols=None, max_cols=None, lmax_n=10, lmax_t="2min"):
        #TODO: Doesn't present combined table properly if evenings enabled
        combi = pd.DataFrame()
        if leq_cols is None:
            leq_cols = [("Leq", "A")]
        if max_cols is None:
            max_cols = [("Lmax", "A")]
        for key in self._logs.keys():
            log = self._logs[key]
            combined_list = []
            # Day
            days = log.leq_by_date(log.get_period(data=log.get_antilogs(), period="days"), cols=leq_cols)
            combined_list.append(days)
            period_headers = ["Daytime" for i in range(len(leq_cols))]
            # Evening
            if log.is_evening():
                evenings = log.leq_by_date(log.get_period(data=log.get_antilogs(), period="evenings"), cols=leq_cols)
                combined_list.append(evenings)
                for i in range(len(leq_cols)):
                    period_headers.append("Evening")
            # Night Leq
            nights = log.leq_by_date(log.get_period(data=log.get_antilogs(), period="nights"), cols=leq_cols)
            combined_list.append(nights)
            for i in range(len(leq_cols)):
                period_headers.append("Night-time")
            # Night max
            maxes = log.as_interval(t=lmax_t)
            maxes = log.get_period(data=maxes, period="nights")
            maxes = log.get_nth_high_low(n=lmax_n, data=maxes)[max_cols]
            combined_list.append(maxes)
            for i in range(len(max_cols)):
                period_headers.append("Night-time")
            summary = pd.concat(objs=combined_list, axis=1)
            summary = self._insert_multiindex(df=summary, super=key)
            combi = pd.concat(objs=[combi, summary], axis=0)
        combi = self._insert_header(df=combi, new_head_list=period_headers, header_idx=0)
        return combi

    def get_modal_l90(self, cols=None, by_date=True, day_t="60min", evening_t="60min", night_t="15min"):
        if cols is None:
            cols = [("L90", "A")]
        combi = pd.DataFrame()
        for key in self._logs.keys():
            # Key is the name of the measurement position
            log = self._logs[key]
            pos_summary = []
            # Daytime
            period_headers = ["Daytime"]
            days = log.get_modal(data=log.get_period(data=log.as_interval(t=day_t), period="days"), by_date=by_date, cols=cols)
            pos_summary.append(days)
            # Evening
            if log.is_evening():
                period_headers.append("Evening")
                evenings = log.get_modal(data=log.get_period(data=log.as_interval(t=evening_t), period="evenings"), by_date=by_date, cols=cols)
                pos_summary.append(evenings)
            # Night time
            nights = log.get_modal(data=log.get_period(data=log.as_interval(t=night_t), period="nights"), by_date=by_date, cols=cols)
            pos_summary.append(nights)
            period_headers.append("Night-time")
            pos_df = pd.concat(pos_summary, axis=1)
            pos_df = self._insert_multiindex(pos_df, super=key)
            combi = pd.concat([combi, pos_df], axis=0)
        combi = self._insert_header(df=combi, new_head_list=period_headers, header_idx=0)
        return combi


    def get_lmax_spectra(self, n=10, t="2min", period="nights"):
        combi = pd.DataFrame()
        # TODO: The night-time timestamp on this is sometimes out by a minute.
        for key in self._logs.keys():
            log = self._logs[key]
            combined_list = []
            maxes = log.get_nth_high_low(n=n, data=log.get_period(data=log.as_interval(t=t), period=period))[["Lmax", "Time"]]
            combined_list.append(maxes)
            summary = pd.concat(objs=combined_list, axis=1)
            summary = self._insert_multiindex(df=summary, super=key)
            combi = pd.concat(objs=[combi, summary], axis=0)
        return combi

    # TODO: get_lowest_l90

    def get_typical_leq_spectra(self, leq_cols=None):
        combi = pd.DataFrame()
        if leq_cols is None:
            leq_cols = ["Leq"]
        for key in self._logs.keys():
            log = self._logs[key]
            combined_list = []
            # Day
            days = log.get_period(data=log.get_antilogs(), period="days")
            days = days[leq_cols].apply(lambda x: np.round(10*np.log10(np.mean(x)), DECIMALS))
            combined_list.append(days)
            period_headers = ["Daytime" for i in range(len(leq_cols))]
            # Evening
            if log.is_evening():
                evenings = log.get_period(data=log.get_antilogs(), period="evenings")
                evenings = evenings[leq_cols].apply(lambda x: np.round(10*np.log10(np.mean(x)), DECIMALS))
                combined_list.append(evenings)
                for i in range(len(leq_cols)):
                    period_headers.append("Evening")
            # Night Leq
            nights = log.get_period(data=log.get_antilogs(), period="nights")
            nights = nights[leq_cols].apply(lambda x: np.round(10*np.log10(np.mean(x)), DECIMALS))
            combined_list.append(nights)
            for i in range(len(leq_cols)):
                period_headers.append("Night-time")
            summary = pd.concat(objs=combined_list, axis=1)
            # summary = self._insert_multiindex(df=summary, super=key)
            combi = pd.concat(objs=[combi, summary], axis=0)
        new_head_dict = {}
        for i in range(len(period_headers)):
            new_head_dict[i] = period_headers[i]
        combi.rename(columns=new_head_dict, inplace=True)
        combi = combi.transpose()
        return combi

# Reporter class is a Work In Progress. This class is intended to take a Survey object
# and export chosen outputs to a Microsoft Word doc.
class Reporter:
    def __init__(self):
        # Initialise the Word file
        self.doc = docx.Document()

    def spectral_table(self, data, heading=None, decimals=0, dba_alignment="left"):
        #TODO: Format the column headers
        # Add the table heading
        assert heading is not None
        self.doc.add_heading(heading, 1)
        # Remove decimal points and 0
        data = data.round(decimals=decimals)
        #TODO: Move A-weighted columns to right

        # Initialise the table
        table = self.doc.add_table(rows=(data.shape[0] + 1), cols=data.shape[1] + 2, style="Table Grid")
        # table.style.TableGrid   # Add in borders
        # Add dates in first column
        table.cell(0, 0).text = "Position"  # Label first column
        table.cell(0, 1).text = "Date"
        ind = data.index.tolist()

        # Loop over the DataFrame and assign data to the Word Table
        # Position names
        for i in range(data.shape[0]):    # For each row
            current_cell = table.cell(i + 1, 0)
            prev_cell = table.cell(i, 0)
            if prev_cell.text == str(ind[i][0]):
                prev_cell.merge(current_cell)
            else:
                current_cell.text = str(ind[i][0])
        #TODO: convert floats to ints

        # Dates
            table.cell(i + 1, 1).text = str(ind[i][1])
            for j in range(data.shape[1]):  # Go through each column
                # Add column headings
                heading = str(data.columns[j][1])  # Remove index params from spectral column headings
                if len(data.columns) == 1:  # This occurs if dba_only=True
                    heading = str(data.columns[j])
                if ".0" in heading:  # Remove decimals from frequency headings, but keep where relevant (31.5)
                    heading = heading.split(".")[0]
                table.cell(0, j + 2).text = heading
                # And assign the values in the table.
                cell = data.iat[i, j]

                # TODO:
    #             Traceback(most
    #             recent
    #             call
    #             last):
    #             File
    #             "C:\Users\tonyr\anaconda3\envs\NoiseSurvey\lib\code.py", line
    #             90, in runcode
    #             exec(code, self.locals)
    #         File
    #         "<input>", line
    #         1, in < module >
    #         File
    #         "C:\Users\tonyr\MSc_Python\NoiseSurvey\PyNS.py", line
    #         494, in spectral_table
    #         if np.isnan(cell):
    #
    # TypeError: ufunc
    # 'isnan'
    # not supported
    # for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''

                try:
                    if np.isnan(cell):
                        cell = "-"
                finally:
                    if isinstance(cell, float) and decimals == 0:
                        cell = int(cell)
                table.cell(i + 1, j + 2).text = str(cell)

    def export(self, path=None, filename="results.docx"):
        # TODO: implement this into Survey object
        assert path is not None
        path = os.path.join(path, filename)
        self.doc.save(path)