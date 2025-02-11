import grpc
import pandas as pd
import forecast_pb2
import forecast_pb2_grpc


def load_and_prepare_data(csv_path: str) -> dict:
    """
    Load time series CSV data and prepare it for the gRPC request.
    Expects the first column to be dates and the second column to be values.
    """
    df = pd.read_csv(csv_path)
    date_col = df.columns[0]
    value_col = df.columns[1]

    # Validate dates and values
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')  # Convert to datetime
    assert not df[date_col].isnull().any(), "Some date values could not be parsed. Check your CSV file."
    assert df[value_col].apply(lambda x: isinstance(x, (int, float))).all(), "All values must be numeric."

    # Sort by date
    df = df.sort_values(by=date_col)
    print(f"First 5 rows of data:\n{df.head()}")  # Debugging output

    return {
        "dates": df[date_col].dt.strftime('%Y-%m-%d').tolist(),
        "values": df[value_col].tolist(),
    }


def get_forecast(stub, data: dict) -> forecast_pb2.ForecastResponse:
    """
    Send a gRPC request to the forecasting server and retrieve the forecast.
    """
    # Set up model parameters
    model_parameters = forecast_pb2.ProphetParameters(
        changepoint_prior_scale=0.08,
        seasonality_prior_scale=12.0,
        holidays_prior_scale=10.0,
        seasonality_mode="multiplicative",
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        growth="linear",  # Change to "logistic" and set cap/floor if needed
        cap=0.0,  # Not used for linear growth
        floor=0.0,  # Not used for linear growth
        n_changepoints=25,
        changepoint_range=0.8,
    )

    # Build the forecast request
    request = forecast_pb2.ForecastRequest(
        data=forecast_pb2.TimeSeriesData(
            dates=data["dates"],
            values=data["values"]
        ),
        periods=30,  # Number of periods to forecast ahead
        model_parameters=model_parameters,
        return_components=True,  # Return trend/seasonality components
    )

    # Send the request to the gRPC server and retrieve the response
    return stub.CreateForecast(request)


def save_forecast(response: forecast_pb2.ForecastResponse, output_path: str):
    """
    Save the forecast results to a CSV file.
    """
    # Create a DataFrame from the forecast response
    df = pd.DataFrame({
        'date': response.forecast_dates,
        'forecast': response.forecast_values,
        'lower_bound': response.forecast_lower_bound,
        'upper_bound': response.forecast_upper_bound,
    })

    # Include any components if provided
    if response.components:
        for component, values in response.components.items():
            df[f'component_{component}'] = values.values

    # Save the DataFrame to a CSV file
    df.to_csv(output_path, index=False)
    print(f"Forecast results saved to {output_path}")


def main():
    # File paths
    csv_path = "Electric_Production.csv"  # Input time series data
    output_path = "forecast_results.csv"  # Output forecast results

    # Load and prepare the input data
    data = load_and_prepare_data(csv_path)

    # Connect to the gRPC server
    channel = grpc.insecure_channel("localhost:50051")  # Update hostname if needed
    stub = forecast_pb2_grpc.ForecastServiceStub(channel)

    # Request a forecast
    try:
        response = get_forecast(stub, data)
        save_forecast(response, output_path)

        # Print some summary information
        print("\nForecast Summary:")
        print(f"Number of periods forecasted: {len(response.forecast_dates)}")
        print(f"First forecast date: {response.forecast_dates[0]}")
        print(f"Last forecast date: {response.forecast_dates[-1]}")
    except grpc.RpcError as e:
        print(f"gRPC error: {e.details()}")


if __name__ == "__main__":
    main()
