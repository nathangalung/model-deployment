from kfp.dsl import pipeline
from kfp.compiler import Compiler
import yaml

# Import components
from components.collect_data import collect_data
from components.preprocess_data import preprocess_data
from components.train_model import train_model
from components.evaluate_model import evaluate_model
from components.deploy_model import deploy_model

import multiprocessing
import psutil

# Load configuration
with open("../config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Auto-generate model name/version if not set
if not config.get("MODEL_NAME"):
    config["MODEL_NAME"] = f"{config['PROJECT_NAME']}-{config['EXPERIMENT_NAME']}".lower().replace(" ", "-")
if not config.get("MODEL_VERSION"):
    config["MODEL_VERSION"] = "1"

# Set max resources if PIPELINE_RESOURCES is empty or missing
if not config.get("PIPELINE_RESOURCES"):
    total_cpu = multiprocessing.cpu_count()
    total_mem_gb = int(psutil.virtual_memory().total / (1024 ** 3))
    config["PIPELINE_RESOURCES"] = {
        "COLLECT_DATA": {
            "MEMORY_REQUEST": f"{total_mem_gb}Gi",
            "MEMORY_LIMIT": f"{total_mem_gb}Gi",
            "CPU_REQUEST": str(total_cpu),
            "CPU_LIMIT": str(total_cpu)
        },
        "PREPROCESS_DATA": {
            "MEMORY_REQUEST": f"{total_mem_gb}Gi",
            "MEMORY_LIMIT": f"{total_mem_gb}Gi",
            "CPU_REQUEST": str(total_cpu),
            "CPU_LIMIT": str(total_cpu)
        },
        "TRAIN_MODEL": {
            "MEMORY_REQUEST": f"{total_mem_gb}Gi",
            "MEMORY_LIMIT": f"{total_mem_gb}Gi",
            "CPU_REQUEST": str(total_cpu),
            "CPU_LIMIT": str(total_cpu)
        },
        "EVALUATE_MODEL": {
            "MEMORY_REQUEST": f"{total_mem_gb}Gi",
            "MEMORY_LIMIT": f"{total_mem_gb}Gi",
            "CPU_REQUEST": str(total_cpu),
            "CPU_LIMIT": str(total_cpu)
        },
        "DEPLOY_MODEL": {
            "MEMORY_REQUEST": "1Gi",
            "MEMORY_LIMIT": "2Gi",
            "CPU_REQUEST": "1",
            "CPU_LIMIT": "2"
        }
    }

@pipeline(
    name=config["PROJECT_NAME"],
    description=f"{config['EXPERIMENT_NAME']} ML Pipeline with Modular Components"
)
def ml_pipeline():
    # Data Collection
    collect_task = collect_data(
        minio_endpoint=config["MINIO"]["ENDPOINT"],
        minio_access_key=config["MINIO"]["ACCESS_KEY"],
        minio_secret_key=config["MINIO"]["SECRET_KEY"],
        minio_bucket=config["MINIO"]["BUCKET"],
        minio_dataset_object=config["MINIO"]["DATASET_OBJECT"]
    )
    collect_resources = config["PIPELINE_RESOURCES"]["COLLECT_DATA"]
    collect_task.set_memory_request(collect_resources["MEMORY_REQUEST"])
    collect_task.set_memory_limit(collect_resources["MEMORY_LIMIT"])
    collect_task.set_cpu_request(collect_resources["CPU_REQUEST"])
    collect_task.set_cpu_limit(collect_resources["CPU_LIMIT"])

    # Data Preprocessing
    preprocess_task = preprocess_data(
        dataset_input=collect_task.outputs['dataset_output'],
        id_columns=config["ID_COLUMNS"],
        target_col=config["TARGET_COL"],
        date_col=config["DATE_COL"],
        ignored_features=config["IGNORED_FEATURES"],
        train_start=config["TRAIN_START_DATE"],
        train_end=config["TRAIN_END_DATE"],
        oot_start=config["OOT_START_DATE"],
        oot_end=config["OOT_END_DATE"]
    )
    preprocess_resources = config["PIPELINE_RESOURCES"]["PREPROCESS_DATA"]
    preprocess_task.set_memory_request(preprocess_resources["MEMORY_REQUEST"])
    preprocess_task.set_memory_limit(preprocess_resources["MEMORY_LIMIT"])
    preprocess_task.set_cpu_request(preprocess_resources["CPU_REQUEST"])
    preprocess_task.set_cpu_limit(preprocess_resources["CPU_LIMIT"])

    # Model Training
    train_task = train_model(
        train_input=preprocess_task.outputs['train_output'],
        oot_input=preprocess_task.outputs['oot_output'],
        feature_selection_report=preprocess_task.outputs['feature_selection_report'],
        target_col=config["TARGET_COL"],
        id_columns=config["ID_COLUMNS"],
        optimization_metric=config["OPTIMIZATION_METRIC"],
        project_name=config["PROJECT_NAME"],
        max_models=config["H2O_CONFIG"]["MAX_MODELS"],
        max_runtime_secs=config["H2O_CONFIG"]["MAX_RUNTIME_SECS"],
        automl_seed=config["H2O_CONFIG"]["AUTOML_SEED"],
        use_cross_validation=config["H2O_CONFIG"]["CROSS_VALIDATION"],
    )
    h2o_config = config["H2O_CONFIG"]
    train_task.set_env_variable("EXPLAINABLE_MODEL", config["EXPLAINABLE_MODEL"])
    train_task.set_env_variable("MAX_MODELS", str(h2o_config["MAX_MODELS"]))
    train_task.set_env_variable("MAX_RUNTIME_SECS", str(h2o_config["MAX_RUNTIME_SECS"]))
    train_task.set_env_variable("USE_CROSS_VALIDATION", h2o_config["CROSS_VALIDATION"])
    train_task.set_env_variable("AUTOML_SEED", str(h2o_config["AUTOML_SEED"]))
    train_resources = config["PIPELINE_RESOURCES"]["TRAIN_MODEL"]
    train_task.set_memory_request(train_resources["MEMORY_REQUEST"])
    train_task.set_memory_limit(train_resources["MEMORY_LIMIT"])
    train_task.set_cpu_request(train_resources["CPU_REQUEST"])
    train_task.set_cpu_limit(train_resources["CPU_LIMIT"])

    # Model Evaluation
    evaluate_task = evaluate_model(
        model_input=train_task.outputs['model_output'],
        train_input=preprocess_task.outputs['train_output'],
        oot_input=preprocess_task.outputs['oot_output'],  # Using OOT as population data starting from OOT onwards
        train_score=train_task.outputs['train_score'],
        oot_score=train_task.outputs['oot_score'],
        optimization_metric=config["OPTIMIZATION_METRIC"]
    ).set_caching_options(False)
    evaluate_resources = config["PIPELINE_RESOURCES"]["EVALUATE_MODEL"]
    evaluate_task.set_memory_request(evaluate_resources["MEMORY_REQUEST"])
    evaluate_task.set_memory_limit(evaluate_resources["MEMORY_LIMIT"])
    evaluate_task.set_cpu_request(evaluate_resources["CPU_REQUEST"])
    evaluate_task.set_cpu_limit(evaluate_resources["CPU_LIMIT"])

    # Model Deployment
    deploy_task = deploy_model(
        evaluation_summary=evaluate_task.outputs['evaluation_summary'],
        model_path=train_task.outputs['model_output'],
        feature_list_path=train_task.outputs['feature_list_output'],
        mlflow_run_id=train_task.outputs['mlflow_run_id'],
        model_name=train_task.outputs['model_name'],
        model_version=train_task.outputs['model_version'],
        kserve_namespace=config["KUBERNETES"]["NAMESPACE"],
        kserve_deployment_mode=config["KSERVE"]["DEPLOYMENT_MODE"]
    )
    deploy_resources = config["PIPELINE_RESOURCES"]["DEPLOY_MODEL"]
    deploy_task.set_memory_request(deploy_resources["MEMORY_REQUEST"])
    deploy_task.set_memory_limit(deploy_resources["MEMORY_LIMIT"])
    deploy_task.set_cpu_request(deploy_resources["CPU_REQUEST"])
    deploy_task.set_cpu_limit(deploy_resources["CPU_LIMIT"])

if __name__ == "__main__":
    Compiler().compile(
        pipeline_func=ml_pipeline,
        package_path="pipeline.yaml"
    )
    print("Modular ML Pipeline compiled successfully: pipeline.yaml")