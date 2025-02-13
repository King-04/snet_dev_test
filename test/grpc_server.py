import grpc
from concurrent import futures
import timeseries_pb2
import timeseries_pb2_grpc
from prophet import Prophet
import pandas as pd

class TimeSeriesService(timeseries_pb2_grpc.TimeSeriesServiceServicer):
    def Forecast(self, request, context):
        try:
            dates_list = list(request.data.dates)
            values_list = list(request.data.values)

            # Prepare the input data
            df = pd.DataFrame({
                'ds': pd.to_datetime(dates_list),
                'y': values_list
            })

            # Configure and train the model
            model_params = request.model_parameters
            model = self.configure_prophet_model(model_params)
            model.fit(df)

            future = model.make_future_dataframe(periods=request.periods)

            # If using logistic growth, set cap and floor for future dates
            if model_params.growth == 'logistic':
                future['cap'] = model_params.cap
                future['floor'] = model_params.floor

            forecast = model.predict(future)

            # response
            response = timeseries_pb2.ForecastResponse(
                forecast_dates=forecast.ds[-request.periods:].dt.strftime('%Y-%m-%d').tolist(),
                forecast_values=forecast.yhat[-request.periods:].tolist(),
                forecast_lower_bound=forecast.yhat_lower[-request.periods:].tolist(),
                forecast_upper_bound=forecast.yhat_upper[-request.periods:].tolist()
            )

        except Exception as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return timeseries_pb2.ForecastResponse()

    def GetDefaultParameters(self, request, context):
        # Return default parameters for the Prophet model
        default_params = timeseries_pb2.ProphetParameters(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            holidays_prior_scale=10.0,
            seasonality_mode="additive",
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            growth="linear",
            n_changepoints=25,
            changepoint_range=0.8
        )
        return default_params

    def configure_prophet_model(self, params):
        model_args = {
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

        # Add capacity parameters for logistic growth
        if params.growth == 'logistic':
            if params.cap is None or params.floor is None:
                raise ValueError("Cap and floor must be specified for logistic growth")
            model_args['growth'] = 'logistic'
            model_args['cap'] = params.cap
            model_args['floor'] = params.floor

        return Prophet(**model_args)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    timeseries_pb2_grpc.add_TimeSeriesServiceServicer_to_server(TimeSeriesService(), server)
    server.add_insecure_port('[::]:50051')
    print("gRPC server running on port 50051...")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()