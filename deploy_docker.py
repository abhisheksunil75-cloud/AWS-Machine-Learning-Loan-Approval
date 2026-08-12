import boto3
import time

# Create a SageMaker client in the AWS Mumbai region
sm_client = boto3.client('sagemaker', region_name='ap-south-1')

# Create an IAM client to access IAM roles
iam_client = boto3.client('iam')

# Create an STS client to retrieve AWS account information
sts_client = boto3.client('sts')

# Get the AWS account ID of the currently authenticated user
account_id = sts_client.get_caller_identity()['Account']

# Define the AWS region
region = 'ap-south-1'


# ---------------------------------------------------------
# 1. Dynamically retrieve the SageMaker execution role
# ---------------------------------------------------------

# Initially, no role ARN is selected
role_arn = None

# Get the list of IAM roles available in the AWS account
roles = iam_client.list_roles()['Roles']

# Search for a role commonly used by SageMaker
for r in roles:

    # Check whether the role name matches either of these patterns
    if 'SageMaker-ExecutionRole' in r['RoleName'] or \
       'SageMakerExecutionRole' in r['RoleName']:

        # Store the ARN of the matching role
        role_arn = r['Arn']

        # Stop searching once a suitable role is found
        break


# If no matching role was found, use a standard SageMaker role ARN format
if not role_arn:

    # Construct the IAM role ARN using the AWS account ID
    role_arn = (
        f"arn:aws:iam::{account_id}:role/"
        f"service-role/AmazonSageMaker-ExecutionRole"
    )


# Display the IAM role being used
print(f"Using IAM Execution Role: {role_arn}")


# ---------------------------------------------------------
# Define SageMaker and ECR resource names
# ---------------------------------------------------------

# URI of the Docker image stored in Amazon ECR
image_uri = (
    f"{account_id}.dkr.ecr.{region}.amazonaws.com/"
    f"ev-loan-prediction:latest"
)

# Name of the SageMaker endpoint
endpoint_name = "ev-loan-prediction-endpoint"

# Name of the SageMaker model
model_name = "ev-loan-docker-model"

# Name of the SageMaker endpoint configuration
config_name = "ev-loan-docker-config"


# ---------------------------------------------------------
# 2. Clean up old SageMaker resources
# ---------------------------------------------------------

print("2. Cleaning up old resources...")


# Delete the old endpoint, endpoint configuration,
# and SageMaker model if they already exist
for delete_fn, kwarg in [
    (sm_client.delete_endpoint, {'EndpointName': endpoint_name}),
    (sm_client.delete_endpoint_config, {'EndpointConfigName': config_name}),
    (sm_client.delete_model, {'ModelName': model_name})
]:

    try:
        # Call the appropriate delete operation
        delete_fn(**kwarg)

        # Wait briefly before continuing
        time.sleep(2)

    except Exception:
        # Ignore errors when the resource does not exist
        pass


# ---------------------------------------------------------
# Wait until the old endpoint is completely removed
# ---------------------------------------------------------

while True:

    try:
        # Try to get information about the old endpoint
        sm_client.describe_endpoint(
            EndpointName=endpoint_name
        )

        # If the endpoint still exists, wait
        print("Waiting for old endpoint cleanup...")
        time.sleep(5)

    except sm_client.exceptions.ClientError:

        # describe_endpoint fails when the endpoint no longer exists
        print("Old endpoint cleared successfully.")

        # Exit the loop
        break


# ---------------------------------------------------------
# 3. Register the Docker image as a SageMaker model
# ---------------------------------------------------------

print("3. Registering SageMaker Model from ECR...")

sm_client.create_model(

    # Name of the SageMaker model
    ModelName=model_name,

    # IAM role that SageMaker uses to access AWS resources
    ExecutionRoleArn=role_arn,

    # Docker container configuration
    PrimaryContainer={
        # Docker image stored in Amazon ECR
        'Image': image_uri
    }
)


# ---------------------------------------------------------
# 4. Create the SageMaker Endpoint Configuration
# ---------------------------------------------------------

print("4. Creating Endpoint Configuration (ml.t2.medium)...")

sm_client.create_endpoint_config(

    # Name of the endpoint configuration
    EndpointConfigName=config_name,

    # Configuration for the production model
    ProductionVariants=[{

        # Name of the traffic variant
        'VariantName': 'AllTraffic',

        # SageMaker model that will receive the requests
        'ModelName': model_name,

        # EC2 instance type used by SageMaker
        'InstanceType': 'ml.t2.medium',

        # Start one instance for the endpoint
        'InitialInstanceCount': 1
    }]
)


# ---------------------------------------------------------
# 5. Create the SageMaker Endpoint
# ---------------------------------------------------------

print("5. Creating Provisioned Endpoint...")

sm_client.create_endpoint(

    # Public name used to identify the endpoint
    EndpointName=endpoint_name,

    # Configuration created in the previous step
    EndpointConfigName=config_name
)


# Deployment request has been submitted
print("Endpoint deployment initiated successfully!")
