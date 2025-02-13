from fastapi import FastAPI, File, UploadFile
from prophet import Prophet
import pandas as pd
import io
import logging

app = FastAPI(title="Prophet Revenue Forecasting Service")


@app.post("/forecast")
async def create_forecast(
        periods: int = 20,
        csv_file: UploadFile = File(...)
):
    """Accepts CSV with 'ds' and 'y' columns"""
    if periods < 1:
        return {"error": "Periods must be ≥ 1"}

    try:
        contents = await csv_file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        if not {'ds', 'y'}.issubset(df.columns):
            return {"error": "CSV requires 'ds' (date) and 'y' (value) columns"}

        df['ds'] = pd.to_datetime(df['ds'])

        model = Prophet()
        model.fit(df)

        # Generate future dates without history
        future = model.make_future_dataframe(
            periods=periods,
            include_history=False,
            freq='W'  # Weekly frequency based on sample data
        )

        forecast = model.predict(future)
        result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

        return {
            "csv": result.to_csv(index=False),
            "message": f"Success: {periods} period forecast"
        }

    except Exception as e:
        logging.error(f"Forecast failed: {str(e)}")
        return {"error": f"Processing error: {str(e)}"}
