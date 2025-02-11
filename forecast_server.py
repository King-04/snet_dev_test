import grpc
from concurrent import futures
import time
import json
import pandas as pd
from prophet import Prophet

# Import the generated classes (ensure forecast_pb2.py and forecast_pb2_grpc.py are in your PYTHONPATH)
import forecast_pb2
import forecast_pb2_grpc

class ForecastServiceServicer(forecast_pb2_grpc.ForecastServiceServicer):
    def CreateForecast(self, request, context):
        try:
            # Debug: Inspect data received
            print(f"Received dates: {request.data.dates[:5]} ... (total: {len(request.data.dates)})")
            print(f"Received values: {request.data.values[:5]} ... (total: {len(request.data.values)})")

            # Convert input to pandas DataFrame
            dates = pd.to_datetime(request.data.dates, errors='coerce')  # Convert to datetime
            if dates.isnull().any():
                raise ValueError("Some dates could not be parsed. Ensure all dates are valid.")

            data = {'ds': dates, 'y': request.data.values}
            df = pd.DataFrame(data).sort_values(by='ds')

            # Validate 'y' column (values)
            if not pd.api.types.is_numeric_dtype(df['y']):
                raise ValueError("'values' column contains non-numeric entries.")

            print(f"DataFrame for Prophet:\n{df.head()}")  # Debugging output

            # Extract model parameters from the request into a dictionary.
            params = request.model_parameters
            model_params = {
                'changepoint_prior_scale': params.changepoint_prior_scale,
                'seasonality_prior_scale': params.seasonality_prior_scale,
                'holidays_prior_scale': params.holidays_prior_scale,
                'seasonality_mode': params.seasonality_mode,
                'yearly_seasonality': params.yearly_seasonality,
                'weekly_seasonality': params.weekly_seasonality,
                'daily_seasonality': params.daily_seasonality,
                'growth': params.growth,
                'n_changepoints': params.n_changepoints,
                'changepoint_range': params.changepoint_range
            }

            # For logistic growth, ensure 'cap' and 'floor' are provided.
            if params.growth == 'logistic':
                if params.cap == 0 or params.floor == 0:
                    context.set_details("For logistic growth, 'cap' and 'floor' must be non-zero.")
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    return forecast_pb2.ForecastResponse()
                model_params['cap'] = params.cap
                model_params['floor'] = params.floor

            # Initialize and train the Prophet model.
            model = Prophet(**model_params)
            model.fit(df)

            # Create future dates for forecasting.
            future = model.make_future_dataframe(periods=request.periods)
            if params.growth == 'logistic':
                future['cap'] = params.cap
                future['floor'] = params.floor

            # Generate forecast.
            forecast = model.predict(future)
            forecast_tail = forecast.tail(request.periods)

            forecast_dates = forecast_tail['ds'].dt.strftime('%Y-%m-%d').tolist()
            forecast_values = forecast_tail['yhat'].tolist()
            forecast_lower_bound = forecast_tail['yhat_lower'].tolist()
            forecast_upper_bound = forecast_tail['yhat_upper'].tolist()

            # Prepare components if requested.
            components = {}
            if request.return_components:
                # Always include trend.
                components['trend'] = forecast_tail['trend'].tolist()
                # Optionally include additional components.
                if 'yearly' in forecast_tail.columns:
                    components['yearly'] = forecast_tail['yearly'].tolist()
                if 'weekly' in forecast_tail.columns:
                    components['weekly'] = forecast_tail['weekly'].tolist()
                if 'daily' in forecast_tail.columns:
                    components['daily'] = forecast_tail['daily'].tolist()

            # Convert components into the gRPC response format.
            components_map = {key: forecast_pb2.FloatList(values=values) for key, values in components.items() if values}

            return forecast_pb2.ForecastResponse(
                forecast_dates=forecast_dates,
                forecast_values=forecast_values,
                forecast_lower_bound=forecast_lower_bound,
                forecast_upper_bound=forecast_upper_bound,
                components=components_map
            )
        except Exception as e:
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return forecast_pb2.ForecastResponse()

    def GetDefaultParameters(self, request, context):
        # Return default Prophet parameters as a JSON string.
        default_params = {
            "changepoint_prior_scale": 0.05,
            "seasonality_prior_scale": 10.0,
            "holidays_prior_scale": 10.0,
            "seasonality_mode": "additive",
            "yearly_seasonality": True,
            "weekly_seasonality": True,
            "daily_seasonality": False,
            "growth": "linear",
            "cap": 0.0,
            "floor": 0.0,
            "n_changepoints": 25,
            "changepoint_range": 0.8
        }
        return forecast_pb2.DefaultParametersResponse(json=json.dumps(default_params))


def serve():
    # Create a gRPC server with a thread pool for handling requests.
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    forecast_pb2_grpc.add_ForecastServiceServicer_to_server(ForecastServiceServicer(), server)
    server.add_insecure_port('[::]:50051')
    print("gRPC server running on port 50051…")
    server.start()
    try:
        # Keep the server running.
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()
