import warnings

warnings.simplefilter("ignore")
#!pip install pystan
#!pip install prophet
# https://gist.github.com/thewisenerd/52f937d01b06287ccf21a05a118e74ad
import pandas as pd
from prophet import Prophet

df = pd.read_csv("dataset.csv")

# some munging
df["Year"] = df["Time Date"].apply(lambda x: str(x)[-4:])
df["Month"] = df["Time Date"].apply(lambda x: str(x)[-6:-4])
df["Day"] = df["Time Date"].apply(lambda x: str(x)[:-6])
df["ds"] = pd.DatetimeIndex(df["Year"] + "-" + df["Month"] + "-" + df["Day"])
df = df.loc[(df["Product"] == 2667437) & (df["Store"] == "QLD_CW_ST0203")]
df.drop(["Time Date", "Product", "Store", "Year", "Month", "Day"], axis=1, inplace=True)
df.columns = ["y", "ds"]
df.head()


m = Prophet(interval_width=0.95, daily_seasonality=True)
model = m.fit(df)

future = m.make_future_dataframe(periods=100, freq="D")
forecast = m.predict(future)
print(forecast.tail())

plot1 = m.plot(forecast)

plt2 = m.plot_components(forecast)
