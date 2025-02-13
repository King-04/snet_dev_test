import requests
import pandas as pd
from io import StringIO


def main():
    print("🕒 Starting Prophet Forecasting Client\n")

    # Get user inputs
    server_url = input("Enter server URL (e.g., http://localhost:8000): ").strip()
    csv_path = input("Path to your CSV file: ").strip()
    periods = input("Number of periods to forecast [default 20]: ").strip()
    periods = int(periods) if periods.isdigit() else 20

    try:
        print(f"\n📤 Sending {csv_path} to {server_url}...")

        with open(csv_path, 'rb') as f:
            response = requests.post(
                f"{server_url}/forecast",
                files={'csv_file': f},
                params={'periods': periods}
            )

        if response.status_code == 200:
            result = response.json()
            if 'csv' in result:
                # Generate output filename
                output_path = csv_path.replace(".csv", f"_forecast_{periods}periods.csv")

                pd.read_csv(StringIO(result['csv'])).to_csv(output_path, index=False)

                print(f"\n✅ Forecast saved to: {output_path}")
                print("Columns included: ds (date), yhat (prediction), yhat_lower/yhat_upper (confidence bounds)")
                return

        print(f"\n❌ Error: {response.json().get('error', 'Unknown error')}")

    except FileNotFoundError:
        print(f"\n❌ File not found: {csv_path}")
    except Exception as e:
        print(f"\n⚠️ Unexpected error: {str(e)}")


if __name__ == "__main__":
    main()
