import grpc
from concurrent import futures
import pandas as pd
from prophet import Prophet

import forecast_server_pb2 as pb2
import forecast_server_pb2_grpc as pb2_grpc


class ForecastServiceServicer(pb2_grpc.ForecastServiceServicer):
    def prepare_data(self, data: pb2.TimeSeriesData) -> pd.DataFrame:
        """Convert proto TimeSeriesData to Prophet's required format."""
        return pd.DataFrame({
            'ds': pd.to_datetime(data.dates),
            'y': data.values
        })

    def configure_prophet_model(self, params: pb2.ProphetParameters) -> Prophet:
        """Configure Prophet model with parameters from proto message."""
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
            if not params.HasField('cap') or not params.HasField('floor'):
                raise ValueError("Cap and floor must be specified for logistic growth")
            model_args['growth'] = 'logistic'
            model_args['cap'] = params.cap
            model_args['floor'] = params.floor

        return Prophet(**model_args)

    def CreateForecast(self, request: pb2.ForecastRequest, context) -> pb2.ForecastResponse:
        """Implementation of the CreateForecast RPC method."""
        try:
            # Prepare the input data
            df = self.prepare_data(request.data)

            # Configure and train the model
            model_params = request.model_parameters if request.HasField('model_parameters') else pb2.ProphetParameters()
            model = self.configure_prophet_model(model_params)
            model.fit(df)

            # Create future dates for forecasting
            future = model.make_future_dataframe(periods=request.periods)

            # Set cap and floor for logistic growth
            if model_params.growth == 'logistic':
                future['cap'] = model_params.cap
                future['floor'] = model_params.floor

            # Make predictions
            forecast = model.predict(future)

            # Prepare the response
            response = pb2.ForecastResponse(
                forecast_dates=forecast.ds[-request.periods:].dt.strftime('%Y-%m-%d').tolist(),
                forecast_values=forecast.yhat[-request.periods:].tolist(),
                forecast_lower_bound=forecast.yhat_lower[-request.periods:].tolist(),
                forecast_upper_bound=forecast.yhat_upper[-request.periods:].tolist()
            )

            # Add components if requested
            if request.return_components:
                components = pb2.ForecastComponents(
                    trend=forecast.trend[-request.periods:].tolist()
                )

                if 'yearly' in forecast:
                    components.yearly.extend(forecast.yearly[-request.periods:].tolist())
                if 'weekly' in forecast:
                    components.weekly.extend(forecast.weekly[-request.periods:].tolist())
                if 'daily' in forecast:
                    components.daily.extend(forecast.daily[-request.periods:].tolist())

                response.components.CopyFrom(components)

            return response

        except Exception as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return pb2.ForecastResponse()

    def GetDefaultParameters(self, request: pb2.Empty, context) -> pb2.ProphetParameters:
        """Implementation of the GetDefaultParameters RPC method."""
        return pb2.ProphetParameters(
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


def serve():
    """Start the gRPC server."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_ForecastServiceServicer_to_server(
        ForecastServiceServicer(), server
    )
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()