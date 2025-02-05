import pandas as pd
import grpc
import forecast_server_pb2 as pb2
import forecast_server_pb2_grpc as pb2_grpc
import numpy as np


def load_and_prepare_data(csv_path: str) -> pb2.ForecastRequest:
    """
    Load time series data from CSV file and convert to gRPC request format.
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Convert dates to datetime and format them
    df['DATE'] = pd.to_datetime(df['DATE'])
    dates = df['DATE'].dt.strftime('%Y-%m-%d').tolist()

    # Convert values to float numpy array then to list
    values = df['IPG2211A2N'].astype(np.float64).tolist()

    print("\nDebug - First few values:")
    for i, (d, v) in enumerate(zip(dates[:5], values[:5])):
        print(f"{i}: {d} ({type(d)}) -> {v} ({type(v)})")

    try:
        # Create TimeSeriesData message step by step
        time_series_data = pb2.TimeSeriesData()
        time_series_data.dates.extend(dates)
        time_series_data.values.extend(values)

        print("\nSuccessfully created TimeSeriesData message")

        # Create ProphetParameters message
        model_parameters = pb2.ProphetParameters(
            changepoint_prior_scale=0.08,
            seasonality_prior_scale=12.0,
            seasonality_mode="multiplicative",
            yearly_seasonality=True,
            growth="linear"
        )

        print("Successfully created ProphetParameters message")

        # Create complete ForecastRequest message
        request = pb2.ForecastRequest(
            data=time_series_data,
            periods=50,
            model_parameters=model_parameters,
            return_components=True
        )

        print("Successfully created ForecastRequest message")

        return request

    except Exception as e:
        print(f"Error creating protobuf message: {str(e)}")
        raise


def get_forecast(request: pb2.ForecastRequest, server_address: str = "localhost:50051"):
    """Send request to the gRPC forecasting service and get predictions"""
    try:
        # Create a gRPC channel
        with grpc.insecure_channel(server_address) as channel:
            # Create a stub (client)
            stub = pb2_grpc.ForecastServiceStub(channel)

            # Make the request
            response = stub.CreateForecast(request)
            return response

    except grpc.RpcError as e:
        print(f"gRPC error: {e.details()}")
        return None


def save_forecast(forecast: pb2.ForecastResponse, output_path: str):
    """Save the forecast results to a CSV file"""
    # Create DataFrame with forecast results
    df = pd.DataFrame({
        'date': forecast.forecast_dates,
        'forecast': forecast.forecast_values,
        'lower_bound': forecast.forecast_lower_bound,
        'upper_bound': forecast.forecast_upper_bound
    })

    # Add components if they exist
    if forecast.HasField('components'):
        components = forecast.components
        if components.trend:
            df['component_trend'] = components.trend
        if components.yearly:
            df['component_yearly'] = components.yearly
        if components.weekly:
            df['component_weekly'] = components.weekly
        if components.daily:
            df['component_daily'] = components.daily

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Forecast saved to {output_path}")


def main():
    csv_path = "Electric_Production.csv"
    output_path = "results.csv"

    try:
        # Load and prepare data
        request = load_and_prepare_data(csv_path)

        if request:
            # Get forecast
            forecast = get_forecast(request)

            if forecast:
                # Save results
                save_forecast(forecast, output_path)

                # Print some basic stats
                print("\nForecast Summary:")
                print(f"Number of periods forecasted: {len(forecast.forecast_dates)}")
                print(f"Last historical date: {forecast.forecast_dates[0]}")
                print(f"Last forecast date: {forecast.forecast_dates[-1]}")

    except Exception as e:
        print(f"Error in main: {str(e)}")


if __name__ == "__main__":
    main()