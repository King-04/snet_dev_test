import pandas as pd
import grpc
import timeseries_pb2
import timeseries_pb2_grpc


def load_and_prepare_data(csv_path: str):
    df = pd.read_csv(csv_path)

    date_col = df.columns[0]
    value_col = df.columns[1]

    # dates to datetime
    df[date_col] = pd.to_datetime(df[date_col], format='%m/%d/%Y', errors='coerce')

    # Drop rows with invalid dates
    df = df.dropna(subset=[date_col])

    data = timeseries_pb2.TimeSeriesData(
        dates=df[date_col].dt.strftime('%Y-%m-%d').tolist(),
        values=df[value_col].tolist()
    )

    # Prepare the ForecastRequest
    request = timeseries_pb2.ForecastRequest(
        data=data,
        periods=30,  # Forecast 30 periods ahead
        model_parameters=timeseries_pb2.ProphetParameters(
            changepoint_prior_scale=0.08,
            seasonality_prior_scale=12.0,
            seasonality_mode="multiplicative",
            yearly_seasonality=True,
            growth="linear"  # Changed to linear as example
        ),
        return_components=True
    )

    return request

def get_forecast(request, grpc_server_address: str = "localhost:50051"):
    """Send request to the gRPC forecasting service and get predictions"""
    try:
        # Create a gRPC channel and stub
        with grpc.insecure_channel(grpc_server_address) as channel:
            stub = timeseries_pb2_grpc.TimeSeriesServiceStub(channel)
            # Call the Forecast method
            response = stub.Forecast(request)
            return response
    except grpc.RpcError as e:
        print(f"Error making gRPC request: {e}")
        return None


def save_forecast(forecast, output_path: str):
    """Save the forecast results to a CSV file"""
    # Create DataFrame with forecast results
    df = pd.DataFrame({
        'date': forecast.forecast_dates,
        'forecast': forecast.forecast_values,
        'lower_bound': forecast.forecast_lower_bound,
        'upper_bound': forecast.forecast_upper_bound
    })

    # Add components if they exist
    if forecast.components:
        for component, values in forecast.components.items():
            df[f'component_{component}'] = values.values

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Forecast saved to {output_path}")


def main():
    csv_path = "Electric_Production.csv"
    output_path = "results.csv"

    # Load and prepare data
    request = load_and_prepare_data(csv_path)

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


if __name__ == "__main__":
    main()