import numpy as np
import pandas as pd
import time

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from epicsarchiver import ArchiverAppliance
from epicsarchiver.retrieval.archiver_retrieval.processor import (
    Processor,
    ProcessorName,
)
from p4p.client.thread import Context

def fetch_history_from_csv(start_time):
    start_time = pd.Timestamp(start_time)

    begin_time = start_time - pd.Timedelta(days=2)
    end_time = start_time

    df = pd.read_csv("./may.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    begin_time = start_time - pd.Timedelta(days=2)
    end_time = start_time

    df = df[
        (df["timestamp"] >= begin_time)
        & (df["timestamp"] < end_time)
    ].copy()

    if df.empty:
        raise RuntimeError(
            f"No data found between {begin_time} and {end_time}"
        )

    df = df[["item_id", "timestamp", "target"]]
    df = df.sort_values(["item_id", "timestamp"])

    return TimeSeriesDataFrame.from_data_frame(
        df,
        id_column="item_id",
        timestamp_column="timestamp",
    )
def fetch_history_from_archiver(start_time):
    end_time = start_time
    begin_time = start_time - pd.Timedelta(days=14)

    archiver = ArchiverAppliance("archiver-linac-01.tn.esss.lu.se")
    my_processor = Processor(processor_name=ProcessorName.MEAN, bin_size=3600)

    events = archiver.get_events(
        "ConVen-G01:HVAC-Ahu-400:TempRoom-R",
        start=begin_time.to_pydatetime(),
        end=end_time.to_pydatetime(),
        processor=my_processor,
    )

    rows = []

    for event in events:
        rows.append(
            {
                "timestamp": pd.to_datetime(event.timestamp).tz_localize(None),
                "target": float(event.val),
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("No archived data was returned")

    df = df.sort_values("timestamp")
    df["item_id"] = "H1"

    return TimeSeriesDataFrame.from_data_frame(
        df,
        id_column="item_id",
        timestamp_column="timestamp",
    )

def write_results_to_epics(ctx, predictions):
    pred = predictions.loc["H1"]

    mean = pred["mean"].to_numpy()
    low = pred["0.1"].to_numpy()
    high = pred["0.9"].to_numpy()

    ctx.put("SYS:ForecastTime", list(range(1, 49)))
    ctx.put("SYS:Forecast", mean[:48])

    # Index 23 is +24h for hourly data
    ctx.put("SYS:24h", float(mean[23]))
    ctx.put("SYS:24hLOW", float(low[23]))
    ctx.put("SYS:24hHIGH", float(high[23]))

    # Index 47 is +48h for hourly data
    ctx.put("SYS:48h", float(mean[47]))
    ctx.put("SYS:48hLOW", float(low[47]))
    ctx.put("SYS:48hHIGH", float(high[47]))

def run_forecast(start_time, predictor, ctx):
    print(f"Running forecast for start time: {start_time}")
    print("Fetching 2 days of archived input data...")
    history = fetch_history_from_csv(start_time)
    #history = fetch_history_from_archiver(start_time)
    print("Running 48 hour prediction...")
    predictions = predictor.predict(history)
    print("Writing forecast to EPICS...")
    write_results_to_epics(ctx, predictions)

def main():
    ctx = Context("pva")

    print("Loading trained AutoGluon predictor...")
    predictor = TimeSeriesPredictor.load("./autogluon-april")

    print(f"Monitoring SYS:StartDateTime ...")

    def on_start_time_change(value):
        try:
            start_time = pd.Timestamp(str(value).strip().split("'")[1])
            run_forecast(start_time, predictor, ctx)
        except Exception as exc:
            print(f"Forecast update failed: {exc}")
            
    subscription = ctx.monitor(
        "SYS:StartDateTime",
        on_start_time_change,
    )
    
    try:
        while True:
            time.sleep(1)
    finally:
        subscription.close()
        ctx.close()

if __name__ == "__main__":
    main()
