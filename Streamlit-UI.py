import os
import tempfile
from typing import List, Dict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import PyNS

# Page configuration
st.set_page_config(page_title="PyNS Acoustic Survey Explorer", layout="wide")

# Sidebar uploader 
st.sidebar.header("Upload CSV files")
files: List[st.runtime.uploaded_file_manager.UploadedFile] = st.sidebar.file_uploader(
    "Select one or more CSV files",
    type="csv",
    accept_multiple_files=True,
)

if not files:
    st.info("Please upload at least one CSV file using the sidebar.")
    st.stop()

# Colour palette 
COLOURS = {
    "Leq A": "#9e9e9e",   # light grey
    "L90 A": "#4d4d4d",   # dark grey
    "Lmax A": "#b41f1f",  # blue
}
TEMPLATE = "plotly"

# Load each CSV into a PyNS Log
logs: Dict[str, PyNS.Log] = {}
for uf in files:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uf.getbuffer())
        path = tmp.name
    try:
        logs[uf.name] = PyNS.Log(path)
    except Exception as err:
        st.error(f"Failed to load {uf.name} into PyNS: {err}")
    finally:
        os.unlink(path)

# Build Survey and pull dataframes
summary_df = leq_spec_df = lmax_spec_df = None
summary_error = ""
if logs:
    try:
        survey = PyNS.Survey()
        if callable(getattr(survey, "add_log", None)):
            for name, lg in logs.items():
                survey.add_log(lg, name=name)
        elif hasattr(survey, "_logs"):
            survey._logs = logs

        summary_df = survey.resi_summary()
        leq_spec_df = getattr(survey, "typical_leq_spectra", lambda: None)()
        lmax_spec_df = getattr(survey, "lmax_spectra", lambda: None)()
    except Exception as err:
        summary_error = str(err)
else:
    summary_error = "No valid logs loaded."

# Helper list of position names
pos_list = [f.name for f in files]

# Helper – tidy spectra style dfs to long format
def spectra_to_rows(df: pd.DataFrame, pos_names: List[str]) -> pd.DataFrame | None:
    if df is None:
        return None
    if not isinstance(df.columns, pd.MultiIndex):
        tidy = df.reset_index().rename(columns={df.index.name or "index": "Period"})
        if "Position" not in tidy.columns:
            tidy.insert(0, "Position", pos_names[0] if pos_names else "Pos1")
        return tidy

    bands = [band for _, band in df.columns][: len({band for _, band in df.columns})]
    set_len = len(bands)
    blocks = []
    for i, pos in enumerate(pos_names):
        start, end = i * set_len, (i + 1) * set_len
        if end > df.shape[1]:
            break
        sub = df.iloc[:, start:end].copy()
        sub.columns = [str(b) for b in bands]
        sub = sub.reset_index().rename(columns={df.index.names[-1] or "index": "Period"})
        if "Position" not in sub.columns:
            sub.insert(0, "Position", pos)
        blocks.append(sub)
    return pd.concat(blocks, ignore_index=True)

# Tabs Resi Summary + one per position
ui_tabs = st.tabs(["Resi Summary"] + pos_list)

#Resi Summary tab
with ui_tabs[0]:
    st.subheader("BS 8233 residential summary")
    if summary_df is not None:
        st.dataframe(summary_df)
    else:
        st.warning(f"Summary unavailable: {summary_error}")

    # Typical Leq and Lmax spectra
    for title, df_data in (("Typical Leq spectra", leq_spec_df), ("Lmax spectra", lmax_spec_df)):
        if df_data is None:
            continue

        tidy = spectra_to_rows(df_data, pos_list).copy()
        tidy["Period"] = tidy["Period"].astype(str)

        st.subheader(title)
        st.dataframe(tidy, hide_index=True)

        # Line graph across octave bands (exclude overall A)
        freq_cols = [c for c in tidy.columns if c not in ("Position", "Period", "A")]
        fig = go.Figure()
        for pos in pos_list:
            for _, row in tidy[tidy["Position"] == pos].iterrows():
                period = row["Period"]
                label = f"{pos} {period}" if len(pos_list) > 1 else period
                mode = "lines+markers" if period.lower().startswith("day") else "lines"
                fig.add_trace(go.Scatter(x=freq_cols, y=row[freq_cols], mode=mode, name=label))
        fig.update_layout(
            template=TEMPLATE,
            title=f"{title} – Day & Night",
            xaxis_title="Octave band (Hz)",
            yaxis_title="dB",
        )
        st.plotly_chart(fig, use_container_width=True)

#Position‑specific tabs 
for tab, uf in zip(ui_tabs[1:], files):
    with tab:
        raw_df = pd.read_csv(uf, parse_dates=[0])
        raw_df.rename(columns={raw_df.columns[0]: "Timestamp"}, inplace=True)

        # Time‑history graph
        if {"Leq A", "L90 A", "Lmax A"}.issubset(raw_df.columns):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=raw_df["Timestamp"], y=raw_df["Leq A"], mode="lines", name="Leq A", line=dict(color=COLOURS["Leq A"], width=1)))
            fig.add_trace(go.Scatter(x=raw_df["Timestamp"], y=raw_df["L90 A"], mode="lines", name="L90 A", line=dict(color=COLOURS["L90 A"], width=1)))
            fig.add_trace(go.Scatter(x=raw_df["Timestamp"], y=raw_df["Lmax A"], mode="markers", name="Lmax A", marker=dict(color=COLOURS["Lmax A"], size=2)))
            fig.update_layout(
                template=TEMPLATE,
                margin=dict(l=0, r=0, t=20, b=80),
                xaxis_title="Timestamp",
                yaxis_title="dB",
                legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="left", x=0),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Required columns ('Leq A', 'L90 A', 'Lmax A') missing.")

        st.subheader("Raw data")
        st.dataframe(raw_df, hide_index=True)
