from kfp.dsl import component, InputPath, OutputPath
from typing import NamedTuple

@component(
    base_image="nathangalung246/kubeflow_dummy:latest"
)
def train_model(
    train_input: InputPath(),
    oot_input: InputPath(),
    feature_selection_report: InputPath(),
    target_col: str,
    id_columns: list,
    optimization_metric: str,
    project_name: str,
    max_models: int,
    max_runtime_secs: int,
    automl_seed: int,
    use_cross_validation: str,
    model_output: OutputPath(),
    feature_list_output: OutputPath()
) -> NamedTuple('Outputs', [
    ('train_score', float),
    ('oot_score', float),
    ('final_features', str),
    ('status', str),
    ('mlflow_run_id', str),
    ('model_name', str),
    ('model_version', str)
]):
    import pandas as pd
    import numpy as np
    import json
    import tempfile
    import shutil
    import os
    from collections import namedtuple
    import h2o
    from h2o.automl import H2OAutoML
    import mlflow
    import mlflow.h2o

    def _train_h2o_model(X_train, y_train, X_oot, y_oot, optimization_metric, max_models, max_runtime_secs, automl_seed, use_cross_validation):
        """H2O AutoML training"""
        # Initialize H2O for wide datasets
        h2o.init(max_mem_size="10G", nthreads=8, strict_version_check=False)

        try:
            # Wide dataset handling
            num_features = len(X_train.columns)
            num_rows = len(X_train)
            
            print(f"Dataset dimensions: {num_rows} rows × {num_features} columns")
            print(f"Memory concern: Wide dataset with {num_features} features")
            
            # Use preprocessed features
            X_train_sample = X_train
            y_train_sample = y_train
            X_oot_sample = X_oot
            y_oot_sample = y_oot
            
            print(f"Using full dataset: {len(X_train_sample)} train rows, {len(X_oot_sample)} OOT rows")
            print(f"Feature count after preprocessing: {len(X_train_sample.columns)} features")

            # Convert to H2O frames
            print("Converting training data to H2O frame...")
            train_data = pd.concat([X_train_sample, y_train_sample], axis=1)
            train_h2o = h2o.H2OFrame(train_data)
            del train_data, X_train_sample, y_train_sample
            
            print("Converting OOT data to H2O frame...")
            oot_data = pd.concat([X_oot_sample, y_oot_sample], axis=1)
            oot_h2o = h2o.H2OFrame(oot_data)
            del oot_data, X_oot_sample, y_oot_sample

            train_h2o[y_train.name] = train_h2o[y_train.name].asfactor()
            oot_h2o[y_oot.name] = oot_h2o[y_oot.name].asfactor()

            # AutoML configuration
            aml = H2OAutoML(
                max_models=max_models,
                max_runtime_secs=max_runtime_secs,
                sort_metric='AUCPR' if optimization_metric.upper() == 'AUCPR' else 'AUC',
                seed=automl_seed,
                nfolds=5 if use_cross_validation.upper() == "YES" else 0,
                # Performance parameters
                max_runtime_secs_per_model=max_runtime_secs // max_models if max_models > 0 else 300,
                stopping_metric='AUCPR' if optimization_metric.upper() == 'AUCPR' else 'AUC',
                stopping_tolerance=0.005,
                stopping_rounds=10,
                balance_classes=True,
                class_sampling_factors=None,
                max_after_balance_size=2.0
            )

            aml.train(x=list(X_train.columns), y=y_train.name, training_frame=train_h2o)

            # Get best model
            best_model = aml.leader

            # Calculate performance
            train_perf = best_model.model_performance(train_h2o)
            oot_perf = best_model.model_performance(oot_h2o)

            if optimization_metric.upper() == 'AUCPR':
                train_score = train_perf.aucpr()
                oot_score = oot_perf.aucpr()
            else:
                train_score = train_perf.auc()
                oot_score = oot_perf.auc()

            return best_model, float(train_score), float(oot_score)
            
        except Exception as e:
            h2o.cluster().shutdown()
            raise e

    def _setup_mlflow():
        """MLflow connection setup"""
        mlflow_endpoints = [
            "http://mlflow-service.kubeflow.svc.cluster.local:5000",
            "http://mlflow-service.kubeflow:5000", 
            "http://10.96.235.221:5000",
            "http://localhost:5000"
        ]
        
        for endpoint in mlflow_endpoints:
            try:
                mlflow.set_tracking_uri(endpoint)
                # Connection test
                mlflow.get_experiment_by_name("Default")
                print(f"MLflow connected successfully to: {endpoint}")
                break
            except Exception as e:
                print(f"Failed to connect to {endpoint}: {e}")
                continue
        else:
            print("Warning: Could not connect to MLflow, using local tracking")
            mlflow.set_tracking_uri("file:///tmp/mlruns")

    # Setup MLflow
    _setup_mlflow()

    def _load_data_with_memory_check(file_path, max_memory_mb=8000):
        """Memory-aware data loading"""
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # File size check
        import os
        file_size_mb = os.path.getsize(file_path) / 1024 / 1024
        print(f"Loading file: {file_size_mb:.1f}MB, Current memory: {initial_memory:.1f}MB")
        
        if initial_memory + file_size_mb > max_memory_mb:
            print(f"Memory limit would be exceeded, loading in chunks")
            # Chunked loading
            chunks = []
            for chunk in pd.read_parquet(file_path, chunksize=10000):
                chunks.append(chunk)
                current_memory = process.memory_info().rss / 1024 / 1024
                if current_memory > max_memory_mb:
                    print(f"Memory limit reached at {current_memory:.1f}MB, processing {len(chunks)} chunks")
                    break
                if current_memory > max_memory_mb * 1.2:  # Emergency stop
                    print(f"EMERGENCY: Memory usage {current_memory:.1f}MB critical - stopping immediately")
                    raise MemoryError(f"Memory usage {current_memory:.1f}MB exceeds emergency threshold")
            df = pd.concat(chunks, ignore_index=True)
            del chunks
            import gc
            gc.collect()
        else:
            df = pd.read_parquet(file_path)
        
        final_memory = process.memory_info().rss / 1024 / 1024
        print(f"Data loaded: {df.shape}, Memory: {final_memory:.1f}MB")
        return df

    # Load datasets with memory monitoring
    print("Loading training data...")
    train_df = _load_data_with_memory_check(train_input)
    print("Loading OOT data...")
    oot_df = _load_data_with_memory_check(oot_input)

    with open(feature_selection_report, 'r') as f:
        feature_report = json.load(f)

    feature_cols = feature_report['selected_features']

    # Prepare data
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_oot = oot_df[feature_cols]
    y_oot = oot_df[target_col]

    print(f"Training model with {len(feature_cols)} features")
    print(f"Train samples: {len(X_train)}, OOT samples: {len(X_oot)}")

    # Start MLflow run
    experiment_name = f"{project_name}-automl-experiment"
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
    except:
        experiment_id = "0"  # Default experiment

    with mlflow.start_run(experiment_id=experiment_id, run_name=f"{project_name}-h2o-automl-run") as run:
        run_id = run.info.run_id
        print(f"MLflow run started: {run_id}")

        # Log parameters
        mlflow.log_param("optimization_metric", optimization_metric)
        mlflow.log_param("max_models", max_models)
        mlflow.log_param("max_runtime_secs", max_runtime_secs)
        mlflow.log_param("automl_seed", automl_seed)
        mlflow.log_param("use_cross_validation", use_cross_validation)
        mlflow.log_param("num_features", len(feature_cols))
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("oot_samples", len(X_oot))

        # Train model
        model, train_score, oot_score = _train_h2o_model(
            X_train, y_train, X_oot, y_oot, optimization_metric,
            max_models, max_runtime_secs, automl_seed, use_cross_validation
        )

        # Log metrics
        mlflow.log_metric(f"train_{optimization_metric.lower()}", train_score)
        mlflow.log_metric(f"oot_{optimization_metric.lower()}", oot_score)
        mlflow.log_metric("train_oot_diff", abs(train_score - oot_score))

        # Save model using H2O's native save method
        model_name = f"h2o-automl-{optimization_metric.lower()}-model"
        
        # Save H2O model - it can be a single file or directory depending on model type
        with tempfile.TemporaryDirectory() as temp_dir:
            # H2O reinitialization
            h2o.init()
            try:
                model_path = h2o.save_model(model=model, path=temp_dir, force=True)
                print(f"H2O model saved to temp path: {model_path}")
                
                # Model path handling
                if os.path.isdir(model_path):
                    # Directory copy
                    shutil.copytree(model_path, model_output, dirs_exist_ok=True)
                else:
                    # File copy
                    shutil.copy2(model_path, model_output)
                
                print(f"H2O model copied to: {model_output}")
                
                # Log model as artifact to MLflow
                try:
                    mlflow.log_artifact(model_path, "model")
                    print("Model artifact logged to MLflow")
                except Exception as e:
                    print(f"Warning: Could not log model artifact to MLflow: {e}")
            finally:
                h2o.cluster().shutdown()

        # Save feature list
        feature_data = {
            "final_features": feature_cols,
            "model_type": "h2o_automl",
            "optimization_metric": optimization_metric,
            "train_score": float(train_score),
            "oot_score": float(oot_score)
        }
        with open(feature_list_output, 'w') as f:
            json.dump(feature_data, f, indent=2)

        # Log feature list as artifact
        try:
            mlflow.log_artifact(feature_list_output, "features")
            print("Feature list logged to MLflow")
        except Exception as e:
            print(f"Warning: Could not log feature list to MLflow: {e}")

        # Log additional metadata
        mlflow.set_tag("model_type", "h2o_automl")
        mlflow.set_tag("project_name", project_name)
        mlflow.set_tag("kubeflow_component", "train_model")

        print(f"Model training complete - {optimization_metric}: Train={train_score:.4f}, OOT={oot_score:.4f}")
        print(f"MLflow run ID: {run_id}")

        # Register model in MLflow Model Registry with auto-increment versioning
        def _register_model_with_auto_increment(model_uri, base_model_name, max_attempts=10):
            """Auto-increment model registration"""
            current_model_name = base_model_name
            attempt = 1
            
            while attempt <= max_attempts:
                try:
                    # Model registration attempt
                    model_version = mlflow.register_model(
                        model_uri=model_uri,
                        name=current_model_name
                    )
                    print(f"Model registered successfully: {current_model_name} v{model_version.version}")
                    return current_model_name, model_version.version
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    
                    # Conflict detection
                    if "already exists" in error_msg or "conflict" in error_msg:
                        # Versioned name fallback
                        current_model_name = f"{base_model_name}-v{attempt}"
                        print(f"Model name conflict, trying: {current_model_name}")
                        attempt += 1
                    else:
                        # Error handling
                        print(f"MLflow registration error: {e}")
                        
                        # Check if it's a model registry unavailable error
                        if "404" in str(e) or "not found" in error_msg:
                            print("MLflow Model Registry not available, using run-based model tracking")
                            return base_model_name, "1"
                        
                        try:
                            # Existing model check
                            client = mlflow.MlflowClient()
                            registered_model = client.get_registered_model(base_model_name)
                            latest_version = registered_model.latest_versions[0].version if registered_model.latest_versions else "1"
                            print(f"Using existing model: {base_model_name} (latest version: {latest_version})")
                            return base_model_name, latest_version
                        except:
                            print("Could not access existing model info, using default")
                            return base_model_name, "1"
            
            # Default fallback
            print(f"Could not register model after {max_attempts} attempts")
            return base_model_name, "1"
        
        try:
            final_model_name, model_version_number = _register_model_with_auto_increment(
                f"runs:/{run_id}/model", 
                model_name
            )
            model_name = final_model_name  # Update model_name to the final registered name
            print(f"Final model registration: {model_name} v{model_version_number}")
        except Exception as e:
            print(f"Warning: Could not register model in MLflow Registry: {e}")
            model_version_number = "1"

        outputs = namedtuple('Outputs', ['train_score', 'oot_score', 'final_features', 'status', 'mlflow_run_id', 'model_name', 'model_version'])
        return outputs(float(train_score), float(oot_score), str(feature_cols), "success", run_id, model_name, model_version_number)

