import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import boto3
import json
import numpy as np
import time
import botocore.exceptions

# Load and prepare the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

# Move the target variable to the first column
iris_df = iris_df[['target'] + [col for col in iris_df.columns if col != 'target']]

train_data, test_data = train_test_split(iris_df, test_size=0.2, random_state=42)

train_data.to_csv('train.csv', index=False, header=False)
test_data.to_csv('test.csv', index=False, header=False)

# Upload the data to Amazon S3
s3 = boto3.resource('s3')
bucket_name = 'yout_bucket_name'
s3.meta.client.upload_file('train.csv', bucket_name, 'iris/train/train.csv')
s3.meta.client.upload_file('test.csv', bucket_name, 'iris/test/test.csv')

# Train a model using SageMaker's built-in XGBoost algorithm
sagemaker = boto3.client('sagemaker', region_name='your_region_name')
role_arn = 'your_role_arn'
xgboost_image = '257758044811.dkr.ecr.us-east-2.amazonaws.com/sagemaker-xgboost:1.2-1' # you can use other algorithms.

training_job_name = f'xgboost-iris-{int(time.time())}'

response = sagemaker.create_training_job(
    TrainingJobName=training_job_name,
    AlgorithmSpecification={
        'TrainingImage': xgboost_image,
        'TrainingInputMode': 'File'
    },
    RoleArn=role_arn,
    InputDataConfig=[
        {
            'ChannelName': 'train',
            'DataSource': {
                'S3DataSource': {
                    'S3DataType': 'S3Prefix',
                    'S3Uri': f's3://{bucket_name}/iris/train/',
                    'S3DataDistributionType': 'FullyReplicated'
                }
            },
            'ContentType': 'text/csv'
        }
    ],
    OutputDataConfig={
        'S3OutputPath': f's3://{bucket_name}/iris/output/'
    },
    ResourceConfig={
        'InstanceType': 'ml.m5.large',
        'InstanceCount': 1,
        'VolumeSizeInGB': 10
    },
    StoppingCondition={
        'MaxRuntimeInSeconds': 3600
    },
    HyperParameters={
        'objective': 'multi:softmax',
        'num_round': '100',
        'num_class': '3'
    }
)

# Wait for the training job to complete
print("Waiting for the training job to complete...")
try:
    sagemaker.get_waiter('training_job_completed_or_stopped').wait(TrainingJobName=training_job_name)
except botocore.exceptions.WaiterError as e:
    print("Waiter encountered an error:", e)

# Check the status of the training job
response = sagemaker.describe_training_job(TrainingJobName=training_job_name)
print("Training job status:", response['TrainingJobStatus'])

if response['TrainingJobStatus'] == 'Failed':
    print("Failure reason:", response['FailureReason'])

# Deploy the model to an endpoint
model_name = f'xgboost-iris-{int(time.time())}'
endpoint_config_name = f'xgboost-iris-endpoint-config-{int(time.time())}'
endpoint_name = f'xgboost-iris-endpoint-{int(time.time())}'

sagemaker.create_model(
    ModelName=model_name,
    PrimaryContainer={
        'Image': xgboost_image,
        'ModelDataUrl': f's3://{bucket_name}/iris/output/{training_job_name}/output/model.tar.gz'
    },
    ExecutionRoleArn=role_arn
)

sagemaker.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[
        {
            'VariantName': 'AllTraffic',
            'ModelName': model_name,
            'InitialInstanceCount': 1,
            'InstanceType': 'ml.m5.large'
        }
    ]
)

sagemaker.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_config_name
)

print(f"Creating endpoint {endpoint_name}...")
waiter = sagemaker.get_waiter('endpoint_in_service')
waiter.wait(EndpointName=endpoint_name)
print(f"Endpoint {endpoint_name} is in service.")

# Make predictions using the deployed model
runtime = boto3.client('sagemaker-runtime', region_name='your_region_name')

sample = test_data.sample(1)
sample_features = sample.drop(columns=['target']).values

response = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType='text/csv',
    Body=','.join(map(str, sample_features.flatten()))
)

result = json.loads(response['Body'].read().decode('utf-8'))
predicted_class = result
true_class = int(sample['target'].values[0])

print(f"Predicted class: {predicted_class}, True class: {true_class}")

# Delete the endpoint
sagemaker.delete_endpoint(EndpointName=endpoint_name)

