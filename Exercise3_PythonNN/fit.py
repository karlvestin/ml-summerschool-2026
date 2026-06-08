import pandas as pd
import matplotlib.pyplot as plt
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

df = pd.read_csv("./april.csv")
df.head()

train_data = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column="item_id",
    timestamp_column="timestamp"
)
train_data.head()

predictor = TimeSeriesPredictor(
    prediction_length=48,
    path="autogluon-april",
    target="target",
    eval_metric="MASE",
)

predictor.fit(
    train_data,
    presets="medium_quality",
    time_limit=600,
)

# Predict first 24h of May based on the april data
predictions = predictor.predict(train_data)
predictions.head()
predictor.plot(train_data, predictions)

# Add May actuals on top
may_data = TimeSeriesDataFrame.from_path("./may.csv")
item_id = predictions.item_ids[0]
target = predictor.target
may_one = may_data.loc[item_id]
ax = plt.gca()
ax.plot(
    may_one.index,
    may_one[target],
    label="May actuals",
    linestyle="--",
)

plt.show()

# Predict may 23:rd based on may 15th to 22nd
train_data = may_data.slice_by_time(
    pd.Timestamp("2026-05-15 00:00:00"),
    pd.Timestamp("2026-05-23 00:00:00"),
)
print(train_data)
predictions = predictor.predict(train_data)
predictions.head()
predictor.plot(train_data, predictions)

# Add May actuals on top
may_data = TimeSeriesDataFrame.from_path("./may.csv")
item_id = predictions.item_ids[0]
target = predictor.target
may_one = may_data.loc[item_id]
ax = plt.gca()
ax.plot(
    may_one.index,
    may_one[target],
    label="May actuals",
    linestyle="--",
)

plt.show()
