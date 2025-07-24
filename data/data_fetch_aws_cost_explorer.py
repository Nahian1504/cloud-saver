import boto3
import pandas as pd
import datetime
import os
import logging


log_file = 'logs/data_fetch_aws_cost_explorer.log'
os.makedirs(os.path.dirname(log_file), exist_ok=True)

logging.basicConfig(
    filename=log_file,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def get_cost_and_usage(start_date, end_date):
    logging.info(f"Fetching cost and usage from {start_date} to {end_date}")
    client = boto3.client('ce', region_name='us-east-1')        # ce = cost explorer

    try:
        response = client.get_cost_and_usage(
            TimePeriod={'Start': start_date, 'End': end_date},
            Granularity='DAILY',
            Metrics=['UnblendedCost', 'UsageQuantity'],
            GroupBy=[{'Type': 'DIMENSION', 'Key': 'SERVICE'}]
        )
    except Exception as e:
        logging.error(f"Failed to fetch data from AWS: {str(e)}")
        raise

    results = []
    for result_by_time in response.get('ResultsByTime', []):
        date = result_by_time['TimePeriod']['Start']
        for group in result_by_time.get('Groups', []):
            service = group['Keys'][0]
            cost = float(group['Metrics']['UnblendedCost']['Amount'])
            usage = float(group['Metrics']['UsageQuantity']['Amount'])
            results.append({'date': date, 'service': service, 'cost': cost, 'usage': usage})

    logging.info(f"Fetched {len(results)} records successfully.")
    return pd.DataFrame(results)

if __name__ == '__main__':
    try:
        today = datetime.date.today()
        start = (today - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
        end = today.strftime('%Y-%m-%d')

        df = get_cost_and_usage(start, end)

        os.makedirs('data', exist_ok=True)
        output_path = os.path.join('data', 'aws_cost_export.csv')
        df.to_csv(output_path, index=False)

        logging.info(f"Data saved to {output_path}")

    except Exception as e:
        logging.critical(f"Script failed with error: {str(e)}")