from concurrent import futures
import grpc
import pandas as pd
from prophet import Prophet
from io import StringIO
import prophet_pb2
import prophet_pb2_grpc
import logging


class ProphetServicer(prophet_pb2_grpc.ProphetForecastServicer):
    def Forecast(self, request, context):
        try:
            # Validate periods
            if request.periods < 1:
                raise ValueError("Periods must be ≥ 1")

            # Read CSV
            df = pd.read_csv(
                StringIO(request.csv_data.decode('utf-8')),
                usecols=['ds', 'y']
            )

            # Validate columns
            if not {'ds', 'y'}.issubset(df.columns):
                raise ValueError("CSV requires 'ds' and 'y' columns")

            # Generate forecast
            model = Prophet()
            model.fit(df)
            future = model.make_future_dataframe(periods=request.periods)
            forecast = model.predict(future)

            return prophet_pb2.ForecastResponse(
                forecast_csv=forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                .to_csv(index=False)
                .encode('utf-8')
            )

        except Exception as e:
            logging.error(f"Forecast failed: {str(e)}")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return prophet_pb2.ForecastResponse(error=str(e))


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    prophet_pb2_grpc.add_ProphetForecastServicer_to_server(ProphetServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("gRPC server running on port 50051")
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
