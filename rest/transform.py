import pandas as pd
import argparse


def transform_csv(input_file, output_file):
    # Read CSV while converting date format
    df = pd.read_csv(
        input_file,
        usecols=[0, 1],  # Keep only first two columns
        parse_dates=['Date'],
        date_parser=lambda x: pd.to_datetime(x, format='%m/%d/%Y')
    )

    # Rename columns and format dates
    df.columns = ['ds', 'y']
    df['ds'] = df['ds'].dt.strftime('%Y-%m-%d')

    # Save transformed data
    df.to_csv(output_file, index=False)
    print(f"✅ Transformation complete! Saved to {output_file}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transform ecommerce CSV data')
    parser.add_argument('input', help='Input CSV file path')
    parser.add_argument('-o', '--output', help='Output CSV file path')

    args = parser.parse_args()
    output_path = args.output or args.input.replace('.csv', '_transformed.csv')

    try:
        transform_csv(args.input, output_path)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
