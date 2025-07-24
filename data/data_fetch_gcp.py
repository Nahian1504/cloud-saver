# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 07:49:59 2025

@author: Asus-PC
"""

import os
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

# Set GCP credentials
KEY_PATH = os.path.join('data', 'gcp-cost-fetcher-228267261a85.json')
credentials = service_account.Credentials.from_service_account_file(KEY_PATH)

# Initialize BigQuery client
project_id = 'gcp-cost-fetcher'  
client = bigquery.Client(credentials=credentials, project=project_id)

# BigQuery SQL query
query = """
SELECT
  usage_start_time,
  usage_end_time,
  project.name AS project_name,
  service.description AS service_description,
  sku.description AS sku_description,
  cost,
  usage.amount AS usage_amount,
  usage.unit AS usage_unit,
  location.location AS region
FROM
  `gcp_billing_export_resource_v1_0118A1_33D74B_A1E208`
WHERE
  _PARTITIONTIME >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
ORDER BY
  usage_start_time DESC
LIMIT 1000
"""


df = client.query(query).to_dataframe()

#Save to CSV
output_path = os.path.join('data', 'gcp_cost_export.csv')
df.to_csv(output_path, index=False)
