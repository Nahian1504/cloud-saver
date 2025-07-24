import boto3
import json
import os
import logging

log_dir = "logs"
log_path = os.path.join(log_dir, "aws_pricing.log")

logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(levelname)s | %(message)s')
console_handler.setFormatter(console_formatter)
logging.getLogger().addHandler(console_handler)

# BOTO3 CLIENT 
try:
    pricing_client = boto3.client('pricing', region_name='us-east-1')
    logging.info("Connected to AWS Pricing API.")
except Exception as e:
    logging.error("Failed to create pricing client: %s", e)
    raise

# FETCH PRICING
def get_aws_pricing(service_code='AmazonEC2', max_results=100, max_pages=5):
    logging.info(f"Fetching pricing data for service: {service_code}")
    paginator = pricing_client.get_paginator('get_products')

    try:
        response_iterator = paginator.paginate(
            ServiceCode=service_code,
            FormatVersion='aws_v1',
            MaxResults=max_results
        )
    except Exception as e:
        logging.error("Pagination error: %s", e)
        raise

    products = []
    for i, page in enumerate(response_iterator):
        if i >= max_pages:
            logging.info(f"Max pages limit reached: {max_pages}")
            break
        logging.info(f"📥 Processing page {i + 1}")
        products.extend(page['PriceList'])

    logging.info(f"Fetched {len(products)} pricing entries.")
    return products

# OUTPUT 
try:
    products = get_aws_pricing()
    output_path = os.path.join("data", "aws_pricing.json")

    
    with open(output_path, 'w') as f:
        json.dump(products, f, indent=2)

    logging.info(f"Saved pricing data to {output_path}")
except Exception as e:
    logging.error("Error during pricing data fetch/save: %s", e)

    
    