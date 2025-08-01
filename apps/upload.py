import kfp
import os
import time
import yaml

def load_config():
    """Load configuration from YAML"""
    config_path = "../config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError("config.yaml not found")
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def connect_kubeflow(endpoint):
    """Connect to Kubeflow client"""
    try:
        client = kfp.Client(host=endpoint)
        print(f"Connected to Kubeflow: {endpoint}")
        return client
    except Exception as e:
        print(f"Connection failed: {e}")
        raise

def upload_pipeline_package(client, config, pipeline_path):
    """Upload or update pipeline"""
    project_name = config["PROJECT_NAME"]
    experiment_name = config["EXPERIMENT_NAME"]
    pipeline_name = f"{project_name} Pipeline"
    pipeline_desc = f"{experiment_name} Kubeflow pipeline"
    
    try:
        # Try upload new pipeline
        pipeline = client.upload_pipeline(
            pipeline_package_path=pipeline_path,
            pipeline_name=pipeline_name,
            description=pipeline_desc
        )
        print(f"Pipeline uploaded: {pipeline.pipeline_id}")
        return pipeline
        
    except Exception as e:
        if "already exists" in str(e):
            # Create new version
            print("Creating new pipeline version...")
            pipeline_version = client.upload_pipeline_version(
                pipeline_package_path=pipeline_path,
                pipeline_version_name=f"v{int(time.time())}",
                pipeline_name=pipeline_name
            )
            print(f"Version created: {pipeline_version.pipeline_version_id}")
            
            # Find existing pipeline
            pipelines = client.list_pipelines()
            for p in pipelines.pipelines:
                if p.display_name == pipeline_name:
                    return p
        else:
            raise e

def create_or_get_experiment(client, config):
    """Create or retrieve experiment"""
    project_name = config["PROJECT_NAME"]
    experiment_name = config["EXPERIMENT_NAME"]
    exp_name = project_name.lower().replace(" ", "-")
    exp_desc = f"{experiment_name} experiment"
    namespace = config.get("KUBERNETES", {}).get("NAMESPACE", "kubeflow")

    try:
        # Try create new experiment
        experiment = client.create_experiment(
            name=exp_name,
            description=exp_desc,
            namespace=namespace
        )
        print(f"Experiment created: {experiment.experiment_id}")
        return experiment

    except Exception as e:
        # Find existing experiment
        print("Finding existing experiment...")
        experiments = client.list_experiments(namespace=namespace)
        for exp in experiments.experiments:
            if exp.display_name == project_name:
                print(f"Using experiment: {exp.experiment_id}")
                return exp
        raise Exception("No experiment found")

def run_pipeline(client, experiment, config, pipeline_path):
    """Execute pipeline run"""
    project_name = config["PROJECT_NAME"]
    job_name = f"{project_name.lower().replace(' ', '-')}-{int(time.time())}"
    
    run = client.run_pipeline(
        experiment_id=experiment.experiment_id,
        job_name=job_name,
        pipeline_package_path=pipeline_path
    )
    
    print(f"Pipeline started: {run.run_id}")
    return run

def validate_minio_config(config):
    """Validate MinIO configuration exists"""
    required_keys = ["MINIO"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing config: {key}")
    
    minio_config = config["MINIO"]
    minio_required = ["ENDPOINT", "ACCESS_KEY", "SECRET_KEY", "BUCKET", "DATASET_OBJECT"]
    for key in minio_required:
        if key not in minio_config:
            raise ValueError(f"Missing MinIO config: {key}")
    
    print("MinIO configuration validated")

def upload_and_run():
    """Main upload and run process"""
    try:
        # Load and validate config
        config = load_config()
        validate_minio_config(config)
        
        # Setup connection
        kubeflow_endpoint = os.getenv('KUBEFLOW_ENDPOINT', 'http://localhost:8891')
        client = connect_kubeflow(kubeflow_endpoint)
        
        # Pipeline package path
        pipeline_path = "pipeline.yaml"
        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"Pipeline not found: {pipeline_path}")
        
        # Upload pipeline
        pipeline = upload_pipeline_package(client, config, pipeline_path)
        
        # Setup experiment
        experiment = create_or_get_experiment(client, config)
        
        # Run pipeline
        run = run_pipeline(client, experiment, config, pipeline_path)
        
        # Display results
        print(f"Pipeline submitted successfully")
        print(f"View run: http://localhost:8088/#/runs/details/{run.run_id}")
        print(f"Experiment: {experiment.experiment_id}")
        print(f"Data source: MinIO ({config['MINIO']['ENDPOINT']})")
        
        return run
        
    except Exception as e:
        print(f"Upload failed: {e}")
        return None

def main():
    """Entry point"""
    print("Starting Kubeflow pipeline upload...")
    print("Using MinIO data source")
    
    run = upload_and_run()
    
    if run:
        print("Upload completed successfully!")
    else:
        print("Upload failed!")
        exit(1)

if __name__ == "__main__":
    main()