from kfp.dsl import component, OutputPath

@component(
    base_image="nathangalung246/kubeflow_dummy:latest",
    packages_to_install=[]
)
def collect_data(
    dataset_output: OutputPath(),
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    minio_bucket: str,
    minio_dataset_object: str
):
    import pandas as pd
    import boto3
    from botocore.client import Config
    import os

    print("Connecting to MinIO...")
    
    try:
        # Connect to MinIO storage
        s3 = boto3.client(
            "s3",
            endpoint_url=f"http://{minio_endpoint}",
            aws_access_key_id=minio_access_key,
            aws_secret_access_key=minio_secret_key,
            config=Config(signature_version="s3v4"),
            region_name="us-east-1"
        )
        print("MinIO client created successfully")

        # Test bucket connection
        try:
            s3.head_bucket(Bucket=minio_bucket)
            print(f"Successfully connected to bucket: {minio_bucket}")
        except Exception as e:
            print(f"Failed to access bucket {minio_bucket}: {e}")
            raise

        # Check object exists
        try:
            s3.head_object(Bucket=minio_bucket, Key=minio_dataset_object)
            print(f"Object {minio_dataset_object} exists in bucket")
        except Exception as e:
            print(f"Object {minio_dataset_object} not found: {e}")
            raise

        # Setup storage configuration
        s3_url = f"s3://{minio_bucket}/{minio_dataset_object}"
        storage_options = {
            "key": minio_access_key,
            "secret": minio_secret_key,
            "client_kwargs": {
                "endpoint_url": f"http://{minio_endpoint}"
            }
        }

        print(f"Reading parquet directly from MinIO: {s3_url}")
        print(f"Storage options: {storage_options}")
        
        # Load and save dataset
        df = pd.read_parquet(s3_url, storage_options=storage_options)
        print(f"Successfully read parquet file: {df.shape}")
        
        df.to_parquet(dataset_output)
        print(f"Dataset collected and saved: {df.shape}")
        
    except Exception as e:
        print(f"Error in collect_data: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise