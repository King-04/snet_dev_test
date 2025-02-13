# grpc_client.py
from io import StringIO

import grpc
import pandas as pd
import prophet_pb2
import prophet_pb2_grpc


def run_client(csv_path, periods, server_address='localhost:50051'):
    try:
        # Read input CSV
        with open(csv_path, 'rb') as f:
            csv_data = f.read()

        # Create channel and stub
        channel = grpc.insecure_channel(server_address)
        stub = prophet_pb2_grpc.ProphetForecastStub(channel)

        # Make request
        response = stub.Forecast(prophet_pb2.ForecastRequest(
            csv_data=csv_data,
            periods=periods
        ))

        # Handle response
        if response.HasField('forecast_csv'):
            output_path = csv_path.replace('.csv', f'_grpc_forecast_{periods}.csv')
            pd.read_csv(StringIO(response.forecast_csv.decode('utf-8'))) \
                .to_csv(output_path, index=False)
            print(f"✅ Forecast saved to {output_path}")
            return True
        else:
            print(f"❌ Error: {response.error}")
            return False

    except Exception as e:
        print(f"⚠️ Client error: {str(e)}")
        return False


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='Path to input CSV')
    parser.add_argument('--periods', type=int, default=20, help='Forecast periods')
    parser.add_argument('--server', default='localhost:50051', help='Server address')
    args = parser.parse_args()

    run_client(args.csv, args.periods, args.server)
