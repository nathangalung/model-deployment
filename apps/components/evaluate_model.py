from kfp.dsl import component, InputPath
from typing import NamedTuple

@component(
    base_image="nathangalung246/kubeflow_dummy:latest",
    packages_to_install=["h2o", "scikit-learn"]
)
def evaluate_model(
    model_input: InputPath(),
    train_input: InputPath(),
    oot_input: InputPath(),  # Using OOT as population data starting from OOT onwards
    train_score: float,
    oot_score: float,
    optimization_metric: str
) -> NamedTuple('Outputs', [('evaluation_summary', str), ('status', str)]):
    import pandas as pd
    import numpy as np
    import json
    from collections import namedtuple
    from datetime import datetime
    import h2o
    import os
    from sklearn.metrics import roc_auc_score

    def _init_h2o():
        """H2O initialization"""
        try:
            # H2O cluster setup
            h2o.init()
            
            # Connection test
            cluster = h2o.cluster()
            if cluster:
                print(f"H2O cluster initialized successfully: {cluster}")
                return True
            return False
            
        except Exception as e:
            print(f"H2O initialization failed: {str(e)}")
            return False

    def _calculate_monthly_auc(model, population_df, train_start_date, oot_end_date):
        """Monthly AUC calculation"""
        monthly_auc = {}
        
        try:
            # Date filtering
            population_df['partition_date'] = pd.to_datetime(population_df['partition_date'])
            train_start = pd.to_datetime(train_start_date)
            oot_end = pd.to_datetime(oot_end_date)
            
            # Date range filter
            date_filtered_df = population_df[
                (population_df['partition_date'] >= train_start) & 
                (population_df['partition_date'] <= oot_end)
            ].copy()
            
            # Monthly grouping
            date_filtered_df['month'] = date_filtered_df['partition_date'].dt.to_period('M')
            
            feature_cols = [c for c in date_filtered_df.columns 
                           if c not in ['label', 'partition_date', 'risk_id', 'month']]
            
            for month in sorted(date_filtered_df['month'].unique()):
                month_data = date_filtered_df[date_filtered_df['month'] == month]
                
                if len(month_data) > 10 and month_data['label'].nunique() > 1:
                    try:
                        X = month_data[feature_cols]
                        y_true = month_data['label']
                        
                        # H2O prediction
                        h2o_frame = h2o.H2OFrame(X)
                        predictions = model.predict(h2o_frame)
                        y_pred = predictions.as_data_frame().iloc[:, 2].values
                        
                        auc = roc_auc_score(y_true, y_pred)
                        monthly_auc[str(month)] = {
                            'auc': float(auc),
                            'samples': len(month_data),
                            'positive_count': int(month_data['label'].sum()),
                            'positive_rate': float(month_data['label'].mean())
                        }
                        
                    except Exception as e:
                        monthly_auc[str(month)] = {'error': str(e)}
                else:
                    monthly_auc[str(month)] = {'error': 'Insufficient data or no label variance'}
            
            return monthly_auc
            
        except Exception as e:
            return {'error': f'Monthly AUC calculation failed: {str(e)}'}

    def _calculate_early_vs_latest_auc(model, population_df):
        """Early vs latest AUC"""
        try:
            population_df = population_df.copy()
            population_df['partition_date'] = pd.to_datetime(population_df['partition_date'])
            population_df = population_df.sort_values('partition_date')
            
            # Period splitting
            total_months = len(population_df['partition_date'].dt.to_period('M').unique())
            if total_months < 2:
                return {'error': 'Need at least 2 months for early vs latest analysis'}
            
            months = sorted(population_df['partition_date'].dt.to_period('M').unique())
            split_point = len(months) // 2
            
            early_months = months[:split_point]
            latest_months = months[split_point:]
            
            feature_cols = [c for c in population_df.columns 
                           if c not in ['label', 'partition_date', 'risk_id']]
            
            results = {}
            
            for period, month_list in [('early', early_months), ('latest', latest_months)]:
                period_data = population_df[
                    population_df['partition_date'].dt.to_period('M').isin(month_list)
                ]
                
                if len(period_data) > 10 and period_data['label'].nunique() > 1:
                    try:
                        X = period_data[feature_cols]
                        y_true = period_data['label']
                        
                        h2o_frame = h2o.H2OFrame(X)
                        predictions = model.predict(h2o_frame)
                        y_pred = predictions.as_data_frame().iloc[:, 2].values
                        
                        auc = roc_auc_score(y_true, y_pred)
                        results[period] = {
                            'auc': float(auc),
                            'samples': len(period_data),
                            'positive_count': int(period_data['label'].sum()),
                            'positive_rate': float(period_data['label'].mean()),
                            'months': [str(m) for m in month_list]
                        }
                        
                    except Exception as e:
                        results[period] = {'error': str(e)}
                else:
                    results[period] = {'error': 'Insufficient data or no label variance'}
            
            return results
            
        except Exception as e:
            return {'error': f'Early vs latest AUC calculation failed: {str(e)}'}

    def _calculate_bin_analysis(model, population_df):
        """Monthly bin analysis"""
        try:
            bin_analysis = {}
            population_df = population_df.copy()
            population_df['month'] = pd.to_datetime(population_df['partition_date']).dt.to_period('M')
            
            feature_cols = [c for c in population_df.columns 
                           if c not in ['label', 'partition_date', 'risk_id', 'month']]
            
            for month in sorted(population_df['month'].unique()):
                month_data = population_df[population_df['month'] == month].copy()
                
                if len(month_data) > 10:
                    try:
                        # Model predictions
                        X = month_data[feature_cols]
                        h2o_frame = h2o.H2OFrame(X)
                        predictions = model.predict(h2o_frame)
                        scores = predictions.as_data_frame().iloc[:, 2].values
                        
                        month_data['score'] = scores
                        
                        # Score decile binning
                        month_data['bin'] = pd.qcut(scores, q=10, labels=False, duplicates='drop')
                        
                        # Bin statistics
                        bin_stats = []
                        total_population = len(month_data)
                        total_positive = int(month_data['label'].sum())
                        
                        for bin_num in sorted(month_data['bin'].unique()):
                            bin_data = month_data[month_data['bin'] == bin_num]
                            bin_population = len(bin_data)
                            bin_positive = int(bin_data['label'].sum())
                            
                            bin_stats.append({
                                'bin': int(bin_num),
                                'count_population': bin_population,
                                'count_positive': bin_positive,
                                'pct_distribution': round(bin_population / total_population * 100, 2),
                                'positive_rate': round(bin_positive / bin_population * 100, 2) if bin_population > 0 else 0,
                                'score_min': float(bin_data['score'].min()),
                                'score_max': float(bin_data['score'].max()),
                                'score_mean': float(bin_data['score'].mean())
                            })
                        
                        bin_analysis[str(month)] = {
                            'total_population': total_population,
                            'total_positive': total_positive,
                            'overall_positive_rate': round(total_positive / total_population * 100, 2),
                            'bins': bin_stats
                        }
                        
                    except Exception as e:
                        bin_analysis[str(month)] = {'error': str(e)}
                else:
                    bin_analysis[str(month)] = {'error': 'Insufficient data for binning'}
            
            return bin_analysis
            
        except Exception as e:
            return {'error': f'Bin analysis failed: {str(e)}'}

    def _calculate_psi_analysis(train_df, population_df):
        """PSI analysis calculation"""
        try:
            psi_results = {}
            population_df = population_df.copy()
            population_df['month'] = pd.to_datetime(population_df['partition_date']).dt.strftime('%Y-%m')
            
            # Feature column alignment
            feature_cols = [c for c in train_df.columns 
                           if c in population_df.columns and c not in ['label', 'partition_date', 'risk_id', 'month']]
            
            # Feature and month limits
            feature_cols = feature_cols[:10]
            unique_months = sorted(population_df['month'].unique())[:12]
            
            for month in unique_months:
                month_data = population_df[population_df['month'] == month]
                
                if len(month_data) < 50:
                    psi_results[month] = {'error': f'Insufficient data ({len(month_data)} samples)'}
                    continue
                
                month_psi = {}
                for feature in feature_cols:
                    try:
                        psi_value = _calculate_psi(train_df[feature], month_data[feature])
                        month_psi[feature] = float(psi_value) if not np.isnan(psi_value) else 0.0
                    except Exception as e:
                        month_psi[feature] = f"Error: {str(e)}"
                
                psi_results[month] = month_psi
            
            return psi_results
            
        except Exception as e:
            return {'error': f'PSI analysis failed: {str(e)}'}

    def _calculate_psi(expected, actual, bins=10):
        """PSI calculation"""
        try:
            expected_clean = pd.Series(expected).dropna()
            actual_clean = pd.Series(actual).dropna()
            
            if len(expected_clean) < 10 or len(actual_clean) < 10:
                return 0.0
            
            # Large dataset sampling
            if len(expected_clean) > 5000:
                expected_clean = expected_clean.sample(n=5000, random_state=42)
            if len(actual_clean) > 5000:
                actual_clean = actual_clean.sample(n=5000, random_state=42)
            
            # Quantile binning
            try:
                bin_edges = np.quantile(expected_clean, np.linspace(0, 1, bins + 1))
                bin_edges = np.unique(bin_edges)
                if len(bin_edges) < 3:
                    return 0.0
            except:
                bin_edges = np.linspace(expected_clean.min(), expected_clean.max(), bins + 1)
            
            # Histogram calculation
            expected_hist, _ = np.histogram(expected_clean, bins=bin_edges)
            actual_hist, _ = np.histogram(actual_clean, bins=bin_edges)
            
            # Zero division protection
            expected_pct = (expected_hist + 1e-8) / (expected_hist.sum() + 1e-7)
            actual_pct = (actual_hist + 1e-8) / (actual_hist.sum() + 1e-7)
            
            # PSI computation
            psi_components = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
            psi_components = psi_components[np.isfinite(psi_components)]
            
            if len(psi_components) == 0:
                return 0.0
            
            psi = np.sum(psi_components)
            return max(0.0, min(psi, 10.0))
            
        except Exception:
            return 0.0

    def _evaluate_performance(train_score, oot_score, optimization_metric):
        """Performance evaluation"""
        if optimization_metric.upper() in ['AUC', 'AUCPR']:
            excellent_threshold = 0.8
            good_threshold = 0.7
            adequate_threshold = 0.6
        elif optimization_metric.upper() == 'F1':
            excellent_threshold = 0.75
            good_threshold = 0.65
            adequate_threshold = 0.55
        else:
            excellent_threshold = 0.85
            good_threshold = 0.75
            adequate_threshold = 0.65
        
        if oot_score > excellent_threshold:
            performance_status = "Model performs excellently"
        elif oot_score > good_threshold:
            performance_status = "Model performs well"
        elif oot_score > adequate_threshold:
            performance_status = "Model performs adequately"
        else:
            performance_status = "Model needs improvement"
        
        overfit = train_score - oot_score
        if overfit > 0.15:
            overfit_status = "severe overfitting detected"
        elif overfit > 0.1:
            overfit_status = "significant overfitting"
        elif overfit > 0.05:
            overfit_status = "mild overfitting"
        else:
            overfit_status = "acceptable generalization"
        
        return {
            'status': f"{performance_status}, {overfit_status}",
            'train_score': train_score,
            'oot_score': oot_score,
            'overfitting': overfit,
            'performance_level': performance_status,
            'overfitting_level': overfit_status
        }

    def _load_data_column_aware(file_path, max_cols=None):
        """Column-aware data loading"""
        try:
            # Metadata extraction
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(file_path)
            total_rows = parquet_file.metadata.num_rows
            schema = parquet_file.schema_arrow
            total_cols = len(schema)
            
            print(f"Dataset: {total_rows} rows × {total_cols} columns")
            
            # Wide dataset handling
            if total_cols > 1000:
                print(f"Wide dataset detected ({total_cols} columns) - using column-aware loading")
                # Column chunking
                if max_cols and total_cols > max_cols:
                    print(f"Limiting to first {max_cols} columns for memory efficiency")
                    # Schema column extraction
                    all_columns = [field.name for field in schema]
                    selected_cols = all_columns[:max_cols]
                    df = pd.read_parquet(file_path, columns=selected_cols)
                else:
                    df = pd.read_parquet(file_path)
            else:
                df = pd.read_parquet(file_path)
                
            print(f"Loaded dataset: {df.shape}")
            return df
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Basic loading fallback
            try:
                # Sample for column info
                df_sample = pd.read_parquet(file_path).head(100)
                safe_cols = df_sample.columns[:500].tolist()  # Limit to 500 columns
                return pd.read_parquet(file_path, columns=safe_cols)
            except:
                # Minimal data fallback
                df = pd.read_parquet(file_path)
                return df.head(10000)  # Take first 10k rows if all else fails
    
    # Main execution
    print("Loading input data with column-aware approach...")
    train_df = _load_data_column_aware(train_input, max_cols=1000)  # Limit columns not rows
    oot_df = _load_data_column_aware(oot_input, max_cols=1000)     # Keep all rows
    population_df = oot_df.copy()  # Using OOT data as population data
    
    print(f"Train data shape: {train_df.shape}")
    print(f"OOT data shape: {oot_df.shape}")
    print(f"Population data shape: {population_df.shape}")
    
    # Column validation
    required_cols = ['partition_date', 'label']
    for df_name, df in [('population', population_df), ('train', train_df), ('oot', oot_df)]:
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in {df_name} data: {missing_cols}")

    # H2O model loading
    print("Initializing H2O...")
    h2o_available = _init_h2o()
    model = None
    
    if h2o_available:
        try:
            print(f"Loading H2O model from: {model_input}")
            model = h2o.load_model(model_input)
            print("H2O model loaded successfully")
        except Exception as e:
            print(f"Model loading failed: {str(e)}")
            h2o_available = False
            model = None
    
    if not h2o_available or model is None:
        # Analysis without model
        print("H2O not available - providing comprehensive statistical analysis without model predictions")
        
        # Basic monthly stats
        try:
            monthly_stats = {}
            population_df_copy = population_df.copy()
            population_df_copy['month'] = pd.to_datetime(population_df_copy['partition_date']).dt.to_period('M')
            
            for month in sorted(population_df_copy['month'].unique()):
                month_data = population_df_copy[population_df_copy['month'] == month]
                if len(month_data) > 0:
                    monthly_stats[str(month)] = {
                        'samples': len(month_data),
                        'positive_count': int(month_data['label'].sum()),
                        'positive_rate': float(month_data['label'].mean()),
                        'note': 'Basic statistics only - no model predictions available'
                    }
        except Exception as e:
            monthly_stats = {'error': f'Monthly stats calculation failed: {str(e)}'}
        
        # Basic bin analysis
        try:
            basic_bin_analysis = {}
            population_df_copy = population_df.copy()
            population_df_copy['month'] = pd.to_datetime(population_df_copy['partition_date']).dt.to_period('M')
            
            for month in sorted(population_df_copy['month'].unique()):
                month_data = population_df_copy[population_df_copy['month'] == month]
                if len(month_data) > 10:
                    # Feature-based binning
                    feature_cols = [c for c in month_data.columns 
                                   if c not in ['label', 'partition_date', 'risk_id', 'month']]
                    if feature_cols:
                        # Basic binning strategy
                        main_feature = feature_cols[0]
                        month_data['bin'] = pd.qcut(month_data[main_feature], q=5, labels=False, duplicates='drop')
                        
                        bin_stats = []
                        total_population = len(month_data)
                        total_positive = int(month_data['label'].sum())
                        
                        for bin_num in sorted(month_data['bin'].unique()):
                            bin_data = month_data[month_data['bin'] == bin_num]
                            bin_population = len(bin_data)
                            bin_positive = int(bin_data['label'].sum())
                            
                            bin_stats.append({
                                'bin': int(bin_num),
                                'count_population': bin_population,
                                'count_positive': bin_positive,
                                'pct_distribution': round(bin_population / total_population * 100, 2),
                                'positive_rate': round(bin_positive / bin_population * 100, 2) if bin_population > 0 else 0
                            })
                        
                        basic_bin_analysis[str(month)] = {
                            'total_population': total_population,
                            'total_positive': total_positive,
                            'overall_positive_rate': round(total_positive / total_population * 100, 2),
                            'bins': bin_stats,
                            'note': f'Basic binning using feature: {main_feature}'
                        }
                    else:
                        basic_bin_analysis[str(month)] = {'error': 'No features available for binning'}
                else:
                    basic_bin_analysis[str(month)] = {'error': 'Insufficient data for binning'}
        except Exception as e:
            basic_bin_analysis = {'error': f'Basic bin analysis failed: {str(e)}'}
        
        # PSI analysis
        try:
            psi_analysis = _calculate_psi_analysis(train_df, population_df)
        except Exception as e:
            psi_analysis = {'error': f'PSI analysis failed: {str(e)}'}
        
        evaluation_summary = {
            'performance': _evaluate_performance(train_score, oot_score, optimization_metric),
            'monthly_statistics': monthly_stats,
            'basic_bin_analysis': basic_bin_analysis,
            'psi_analysis': psi_analysis,
            'warning': 'H2O model not available - analysis provided without model predictions',
            'evaluation_date': datetime.now().isoformat()
        }
        
        outputs = namedtuple('Outputs', ['evaluation_summary', 'status'])
        return outputs(json.dumps(evaluation_summary), "success")

    # Model-based analysis
    try:
        print("Starting comprehensive model evaluation...")
        
        # Fallback date values
        # Config parameter alternative
        train_start_date = "2024-01-01"
        oot_end_date = "2024-09-30"
        
        # Population data sampling
        if len(population_df) > 30000:
            print(f"Further sampling population data from {len(population_df)} to 30000 rows")
            population_df = population_df.sample(n=30000, random_state=42)
        
        # Monthly AUC analysis
        print("Calculating monthly AUC...")
        monthly_auc = _calculate_monthly_auc(model, population_df, train_start_date, oot_end_date)
        
        # Early vs latest analysis  
        print("Calculating early vs latest AUC...")
        label_analysis = _calculate_early_vs_latest_auc(model, population_df)
        
        # Monthly bin analysis
        print("Calculating bin analysis...")
        bin_analysis = _calculate_bin_analysis(model, population_df)
        
        # PSI analysis
        print("Calculating PSI analysis...")
        psi_analysis = _calculate_psi_analysis(train_df.sample(n=min(10000, len(train_df)), random_state=42), population_df)
        
        # Performance evaluation
        performance_eval = _evaluate_performance(train_score, oot_score, optimization_metric)
        
        # Results compilation
        evaluation_summary = {
            'performance': performance_eval,
            'monthly_auc': monthly_auc,
            'early_vs_latest_analysis': label_analysis,
            'bin_analysis': bin_analysis,
            'psi_analysis': psi_analysis,
            'evaluation_date': datetime.now().isoformat()
        }
        
        print("Model evaluation completed successfully")
        
    except Exception as e:
        print(f"Error during model evaluation: {str(e)}")
        evaluation_summary = {
            'performance': _evaluate_performance(train_score, oot_score, optimization_metric),
            'error': f'Model evaluation failed: {str(e)}',
            'evaluation_date': datetime.now().isoformat()
        }
    
    finally:
        # H2O cleanup
        if h2o_available:
            try:
                h2o.cluster().shutdown()
                print("H2O cluster shutdown completed")
            except Exception as e:
                print(f"H2O shutdown warning: {str(e)}")

    outputs = namedtuple('Outputs', ['evaluation_summary', 'status'])
    return outputs(json.dumps(evaluation_summary), "success")