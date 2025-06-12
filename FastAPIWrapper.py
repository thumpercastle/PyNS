from fastapi import FastAPI, Request, Response, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from PyNS import Log, Survey
from typing import Optional
from fastapi.responses import RedirectResponse
from uuid import uuid4
from pydantic import BaseModel
import pandas as pd
import numpy as np
import io
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend domain(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#class GetResRequest(BaseModel):
#    session_id: str = "default"

class SetPeriodsRequest(BaseModel):
    session_id: str = "default"
    times: Optional[list] = None  # List of time periods to set, e.g., ["day", "evening", "night"]


# In-memory store for Survey objects, keyed by session or user id (for demo, use a global singleton)
survey_store = {}

@app.post("/survey/logs")
async def survey_logs(
    file: UploadFile = File(...),
    name: str = Form("name"),
    session_id: str = Form("session_id")
):
    """
    Add a Log object to the Survey object.
    :param data: Initialised Log object
    :param name: Name of the position, e.g. "A1"
    :return: None.
    """
    print(f"Adding log: {file.filename} with name: {name} for session_id: {session_id}")
    survey = survey_store.get(session_id)
    if survey is None:
        print(f"Creating new survey for session_id: {session_id}")
        survey = Survey()
        survey_store[session_id] = survey
    try:
        contents = await file.read()
        print ("contents")
        df = pd.read_csv(io.StringIO(contents.decode()), index_col="Time", parse_dates=["Time"], dayfirst=True)
        print ("read")
        log = Log("", df)
        print ("log created")
        survey.add_log(log, name)
        print(f"Log {name} added successfully for session {session_id}")
        return {"success": True, "message": f"Log {name} added successfully for session {session_id}"}        
    except Exception as e:
        return {"error": f"Failed to read log file: {str(e)}"}

@app.put("/survey/periods")
def set_periods(req: SetPeriodsRequest):
    """
        Set the daytime, evening and night-time periods of all Log objects in the Survey.
        To disable evening periods, simply set it the same as night-time.
        :param times: A dictionary with strings as keys and integer tuples as values.
        The first value in the tuple represents the hour of the day that period starts at (24hr clock), and the
        second value represents the minutes past the hour.
        e.g. for daytime from 07:00 to 19:00, evening 19:00 to 23:00 and night-time 23:00 to 07:00,
        times = {"day": (7, 0), "evening": (19, 0), "night": (23, 0)}
        NOTES:
        Night-time must cross over midnight. (TBC experimentally).
        Evening must be between daytime and night-time. To
        :return: None.
    """
    survey = survey_store.get(req.session_id)
    if survey is None:
        return {"error": f"No survey found for {req.session_id} . Please add a log first."}
    # Todo need to pass a list of times
    if not req.times:
        return {"error": "No times provided. Please specify periods to set."}
    survey.set_periods(times=req.times)
    return {"status": "periods set", "session_id": req.session_id}

@app.get("/survey/periods")
def get_periods(session_id: str):
    """
    Check the currently-set daytime, evening and night-time periods for each Log object in the Survey.
    :return: Tuples of start times.
    """    
    survey = survey_store.get(session_id)
    if survey is None:
        return {"error": "No survey found for this session. Please add a log first."}

    result = survey.get_periods()
    if isinstance(result, pd.DataFrame):
        result = CleanDataFrame(result)
        return result
    else:
        return {"error": f"Expected DataFrame, got {type(result)}"}

@app.get("/survey/summary")
def get_resi_summary(session_id: str):
        """
        Get a dataframe summarising the parameters relevant to assessment of internal ambient noise levels in
        UK residential property assessments. Daytime and night-time Leqs, and nth-highest Lmax values all presented
        in a succinct table. These will be summarised as per the daytime, evening and night-time periods set (default
        daytime 07:00 to 23:00 and night-time 23:00 to 07:00).
        The date of the Lmax values are presented for the night-time period beginning on that date. i.e. an Lmax
        on 20/12/2024 would have occurred in the night-time period starting on that date and ending the following
        morning.
        :param leq_cols: List of tuples. The columns on which to perform Leq calculations. This can include L90
        columns, or spectral values. e.g.  leq_cols = [("Leq", "A"), ("L90", "125")]
        :param max_cols: List of tuples. The columns on which to get the nth-highest values.
        Default max_cols = [("Lmax", "A")]
        :param lmax_n: Int. The nth-highest value for max_cols. Default 10 for 10th-highest.
        :param lmax_t: String. This is the time period over which to compute nth-highest Lmax values.
        e.g. "2min" computes the nth-highest Lmaxes over 2-minute periods. Note that the chosen period must be
        equal to or more than the measurement period. So you cannot measure in 5-minute periods and request 2-minute
        Lmaxes.
        :return: A dataframe presenting a summary of the Leq and Lmax values requested.
        """
    print(f"Processing survey_resi_summary for session_id: {session_id}")

    survey = survey_store.get(session_id)

    if survey is None:
        return {"error": "No survey found for this session. Please add a log first."}

    result = survey.resi_summary()

    if isinstance(result, pd.DataFrame):
        result = CleanDataFrame(result)
        return result
    else:
        return {"error": f"Expected DataFrame, got {type(result)}"}



@app.get("/survey/modal_l90")
def survey_modal_l90(session_id: str):
    """
    Get a dataframe summarising Modal L90 values for each time period, as suggested by BS 4142:2014.
    Currently, this method rounds the values to 0 decimal places by default and there is no alternative
    implementation.
    Note that this function will estimate L90s as a longer value by performing an Leq computation on them.
    The measured data in Logs must be smaller than or equal to the desired period, i.e. you can't measure in 15-
    minute periods and request 5-minute modal values.
    :param cols: List of tuples of the columns desired. This does not have to be L90s, but can be any column.
    :param by_date: Bool. If True, group the modal values by date. If False, present one modal value for each
    period.
    :param day_t:  String. Measurement period T. i.e. daytime measurements will compute modal values of
    L90,60min by default.
    :param evening_t: String. Measurement period T. i.e. evening measurements will compute modal values of
    L90,60min by default, unless evenings are disabled (which they are by default).
    :param night_t: Measurement period T. i.e. night-time measurements will compute modal values of
    L90,15min by default.
    :return: A dataframe of modal values for each time period.
    """
  

    survey = survey_store.get(session_id)
    if survey is None:
        return {"error": "No survey found for this session. Please add a log first."}
    result = survey.modal_l90(cols=None, by_date=True, day_t="60min", evening_t="60min", night_t="15min")

    return result.to_dict()



@app.get("/survey/lmax_spectra")
def survey_lmax_spectra(session_id: str):

    """
    Get spectral data for the nth-highest Lmax values during a given time period.
    This computes Lmax Event spectra. Lmax Hold spectra has not yet been implemented.
    Assumptions and inputs as per Survey.resi_summary() method.
    IMPORTANT: The dates of the Lmax values are presented for the night-time period beginning on that date.
    This means that for early morning timings, the date is behind by one day.
    e.g. an Lmax presented as occurring at 20/12/2024 at 01:22 would have occurred at 21/12/2024 at 01:22.
    :param n: Int. Nth-highest Lmax. Default 10th-highest.
    :param t: String. This is the time period over which to compute nth-highest Lmax values.
    e.g. "2min" computes the nth-highest Lmaxes over 2-minute periods. Note that the chosen period must be
    equal to or more than the measurement period. So you cannot measure in 5-minute periods and request 2-minute
    Lmaxes.
    :param period: String. "days", "evenings" or "nights"
    :return: Dataframe of nth-highest Lmax Event spectra.
    """


    survey = survey_store.get(session_id)
    if survey is None:
        return {"error": "No survey found for this session. Please add a log first."}
    result = survey.lmax_spectra(n=10, t="2min", period="night")
    return result.to_dict()


@app.get("/survey/typical_leq_spectra")
def survey_typical_leq_spectra(session_id: str):
    """
    Compute Leqs over daytime, evening and night-time periods.
    This is an overall Leq, and does not group Leqs by date.
    :param leq_cols: List of strings or List of Tuples.
    For all Leq columns, use ["Leq"]. For specific columns, use list of tuples [("Leq", "A"), ("Leq", 125)]
    :return: A dataframe with a continuous Leq computation across dates, for each time period.
    """
    survey = survey_store.get(session_id)
    if survey is None:
        return {"error": "No survey found for this session. Please add a log first."}
    result = survey.typical_leq_spectra(leq_cols=None)
    #return result.to_dict(orient='records')
    return result.to_dict(orient='list')


# Serve static files (including test_api.html) from the current directory
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
def root():
    return RedirectResponse(url="/static/test_api.html")


def CleanDataFrame(result: pd.DataFrame) -> dict:
    
    """
    Clean a DataFrame for JSON serialization:
    """
    # Convert all column names to strings
    result.columns = [str(col) for col in result.columns]
    # Reset index if it's not a simple RangeIndex
    if not isinstance(result.index, pd.RangeIndex):
        result = result.reset_index()
    # Replace NaN/inf/-inf values with None for JSON compliance
    result = result.replace([np.nan, np.inf, -np.inf], None)

    return {"data": result.to_dict(orient='records')}

# --- MCP ENDPOINTS ---
@app.get("/v1/health")
def mcp_health():
    return {"status": "ok"}

@app.get("/v1/models")
def mcp_models():
    log_methods = [m for m in dir(Log) if not m.startswith('_') and callable(getattr(Log, m))]
    survey_methods = [m for m in dir(Survey) if not m.startswith('_') and callable(getattr(Survey, m))]
    return {
        "object": "list",
        "data": [
            {"id": f"log.{m}", "object": "model"} for m in log_methods
        ] + [
            {"id": f"survey.{m}", "object": "model"} for m in survey_methods
        ]
    }

@app.post("/v1/completions")
async def mcp_completions(request: Request):
    body = await request.json()
    model = body.get("model")
    input_data = body.get("input", {})
    # Parse model string
    if model and model.startswith("log."):
        method = model.split(".", 1)[1]
        obj = Log(**input_data.get("init", {}))
    elif model and model.startswith("survey."):
        method = model.split(".", 1)[1]
        # Use session_id if provided, else create new Survey
        session_id = input_data.get("session_id")
        if session_id and session_id in survey_store:
            obj = survey_store[session_id]
        else:
            obj = Survey(**input_data.get("init", {}))
            if session_id:
                survey_store[session_id] = obj
    else:
        return {"error": "Unknown model"}
    # Call the method
    func = getattr(obj, method, None)
    if not func or not callable(func):
        return {"error": f"Method {method} not found"}
    params = input_data.get("params", {})
    try:
        output = func(**params) if params else func()
    except Exception as e:
        return {"error": str(e)}
    # If output is a DataFrame, convert to records
    try:
        import pandas as pd
        import numpy as np
        if isinstance(output, pd.DataFrame):
            output = output.replace([np.nan, np.inf, -np.inf], None)
            output = output.to_dict(orient='records')
    except Exception:
        pass
    return {
        "object": "completion",
        "model": model,
        "output": output
    }