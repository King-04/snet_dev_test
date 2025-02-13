from fastapi import FastAPI, File, UploadFile
from prophet import Prophet
import pandas as pd
import io
import logging

app = FastAPI(title="Prophet Forecasting Service")


@app.post("/forecast")
async def create_forecast(
        periods: int = 20,
        csv_file: UploadFile = File(...)
):
    """Accepts CSV with 'ds' and 'y' columns, returns forecast"""

    # Validate input
    if periods < 1:
        return {"error": "Periods must be ≥ 1"}

    try:
        # Read CSV
        contents = await csv_file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        # Validate columns
        if not {'ds', 'y'}.issubset(df.columns):
            return {"error": "CSV requires 'ds' (date) and 'y' (value) columns"}

        # Train model
        model = Prophet()
        model.fit(df)

        # Generate future dates
        future = model.make_future_dataframe(periods=periods)

        # Make predictions
        forecast = model.predict(future)

        # Return forecast subset
        result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        return {
            "csv": result.to_csv(index=False),
            "message": f"Success: {periods} period forecast"
        }

    except Exception as e:
        logging.error(f"Forecast failed: {str(e)}")
        return {"error": f"Processing error: {str(e)}"}
