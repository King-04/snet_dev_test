import grpc
import timeseries_pb2
import timeseries_pb2_grpc


def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = timeseries_pb2_grpc.TimeSeriesServiceStub(channel)

        # Hardcoded valid data
        data = timeseries_pb2.TimeSeriesData(
            dates=["1985-01-01", "1985-02-01", "1985-03-01"],
            values=[72.5052, 70.672, 62.4502]
        )

        # Prepare the request
        request = timeseries_pb2.ForecastRequest(
            data=data,
            periods=3,
            model_parameters=timeseries_pb2.ProphetParameters(
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
                seasonality_mode="additive",  # Explicitly set seasonality_mode
                growth="linear"
            ),
            return_components=True
        )

        # Send the request
        try:
            response = stub.Forecast(request)
            print("Forecast successful!")
            print("Forecast dates:", response.forecast_dates)
            print("Forecast values:", response.forecast_values)
            if response.components:
                print("Components:")
                for component, values in response.components.items():
                    print(f"{component}: {values.values}")
        except grpc.RpcError as e:
            print(f"gRPC error: {e.code()}: {e.details()}")


if __name__ == "__main__":
    run()