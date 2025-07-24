import boto3
import pandas as pd
import logging
import os
from datetime import datetime, timezone


log_dir = "logs"
log_file = os.path.join(log_dir, f"spot_instance_fetch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# Fetch Spot Instance Data 
def fetch_spot_instance_lifetimes(region='us-east-1'):
    logging.info("Fetching spot instance requests...")
    ec2 = boto3.client('ec2', region_name=region)
    data = []

    try:
        response = ec2.describe_spot_instance_requests()
    except Exception as e:
        logging.error(f"Error describing spot requests: {e}")
        return

    for req in response['SpotInstanceRequests']:
        request_id = req['SpotInstanceRequestId']
        state = req['State']
        status_code = req['Status']['Code']
        instance_id = req.get('InstanceId')

        if not instance_id:
            logging.info(f"Skipping request {request_id} with no instance assigned.")
            continue

        try:
            instance_data = ec2.describe_instances(InstanceIds=[instance_id])
            instance = instance_data['Reservations'][0]['Instances'][0]

            launch_time = instance['LaunchTime']
            now = datetime.now(timezone.utc)
            interrupted = 1 if 'instance-terminated' in status_code.lower() else 0

            # Estimate duration
            if instance['State']['Name'] in ['terminated', 'stopped']:
                end_time = now
            else:
                end_time = now

            duration = end_time - launch_time
            duration_hours = round(duration.total_seconds() / 3600, 2)

            data.append({
                'instance_id': instance_id,
                'duration_hours': duration_hours,
                'interrupted': interrupted
            })

            logging.info(f"Instance {instance_id} | Duration: {duration_hours} hrs | Interrupted: {interrupted}")

        except Exception as e:
            logging.warning(f"Could not fetch data for instance {instance_id}: {e}")

    if not data:
        logging.warning("No valid spot instance data retrieved.")
        return

   
    df = pd.DataFrame(data)
    output_path = os.path.join('data', 'aws_real_spot_instances.csv')
    df.to_csv(output_path, index=False)

    logging.info(f"CSV saved to {output_path}")
    logging.info(f"Log saved to {log_file}")


fetch_spot_instance_lifetimes()