# ML Ops

See notebook arima_forecast.ipynb

from statsmodels.tsa.statespace.sarimax import SARIMAX

1. Gather Data
2. If not stationary - apply transformations
3. d = number of times series is differenced
4. List values of p and q
5. Fit every combination in a loop - 
6. Select model with lowest AIC
7. Observe residual analysis
8. If uncorrelated residuals ... we're ready for forecasting!

Using Daloopa to pull data for AMZN:

![image](https://user-images.githubusercontent.com/39496491/224386099-fb8937b0-cca5-4597-ad1e-08e47fd16a56.png)

We isolate Net Sales

![image](https://user-images.githubusercontent.com/39496491/224386278-4331d134-4625-423d-9b79-70dd936ee802.png)

And difference it twice to remove the trend and seasonality:

![image](https://user-images.githubusercontent.com/39496491/224386367-14031c91-29dd-411e-a622-ef4b029e9254.png)

our ARIMA looks ok:

![image](https://user-images.githubusercontent.com/39496491/224386457-c9cf777d-5cc2-488a-be57-b2812f599623.png)

The ARIMA model looks better than naive seasonal:

![image](https://user-images.githubusercontent.com/39496491/224387092-bed3d10c-47e0-4069-9ff5-3193bb1751d9.png)

and the MAPE (% error) is better:

![image](https://user-images.githubusercontent.com/39496491/224387272-b449121a-154c-441e-97f0-7a80066ef879.png)

So - this model appears to be useful for forecasting.

