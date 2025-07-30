from kfp.dsl import component, InputPath, OutputPath
from typing import NamedTuple

@component(
    base_image="nathangalung246/kubeflow_dummy:latest"
)
def preprocess_data(
    dataset_input: InputPath(),
    train_output: OutputPath(),
    oot_output: OutputPath(),
    feature_selection_report: OutputPath(),
    id_columns: list,
    target_col: str,
    date_col: str,
    ignored_features: list,
    train_start: str,
    train_end: str,
    oot_start: str,
    oot_end: str
) -> NamedTuple('Outputs', [('train_shape', str), ('oot_shape', str), ('selected_features', str), ('status', str)]):
    import pandas as pd
    import numpy as np
    from collections import namedtuple
    import json
    import lightgbm as lgb
    import gc
    import psutil
    import pyarrow.parquet as pq
    
    def _optimize_dtypes(df):
        """Memory optimization for DataFrame"""
        print(f"Memory usage before optimization: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != 'object':
                c_min = df[col].min()
                c_max = df[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                        
                elif str(col_type)[:5] == 'float':
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
        
        print(f"Memory usage after optimization: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        return df
    
    def _monitor_memory():
        """Memory usage monitoring"""
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        # Memory thresholds - adjusted for 5GB pod limit
        warning_threshold = 3000   # 3GB warning  
        critical_threshold = 4500  # 4.5GB critical (within 5GB pod limit)
        
        if memory_mb > critical_threshold:
            print(f"CRITICAL: Memory usage {memory_mb:.2f} MB exceeds {critical_threshold} MB - terminating to prevent system crash")
            raise MemoryError(f"Memory usage {memory_mb:.2f} MB exceeds critical threshold {critical_threshold} MB")
        elif memory_mb > warning_threshold:
            print(f"WARNING: High memory usage {memory_mb:.2f} MB (threshold: {warning_threshold} MB)")
        else:
            print(f"Current memory usage: {memory_mb:.2f} MB")
        
        return memory_mb

    def _validate_data(df, id_columns, target_col, date_col, train_start, train_end, oot_start, oot_end):
        """Dataset validation checks"""
        
        # Check duplicates using ID + target
        check_cols = id_columns + [target_col]
        duplicates = df[check_cols].duplicated().sum()
        if duplicates > 0:
            raise ValueError(f"Found {duplicates} duplicate rows")
        
        # Check target has only 2 values
        unique_targets = df[target_col].nunique()
        if unique_targets != 2:
            raise ValueError(f"Target must have 2 values, found {unique_targets}")
        
        # Check date range
        df[date_col] = pd.to_datetime(df[date_col])
        min_date = df[date_col].min()
        max_date = df[date_col].max()
        
        # Check if we have data for training period
        if min_date > pd.to_datetime(train_start) or max_date < pd.to_datetime(train_end):
            raise ValueError(f"Data range {min_date} to {max_date} doesn't cover training period {train_start} to {train_end}")
        
        # Check if we have some OOT data (warn if partial)
        oot_start_dt = pd.to_datetime(oot_start)
        oot_end_dt = pd.to_datetime(oot_end)
        
        if max_date < oot_start_dt:
            raise ValueError(f"No OOT data available. Data ends at {max_date} but OOT starts at {oot_start}")
        elif max_date < oot_end_dt:
            print(f"WARNING: OOT period is partially covered. Data ends at {max_date} but OOT should end at {oot_end}")
        
        # Check column names are alphanumeric + underscore
        for col in df.columns:
            if not col.replace('_', '').isalnum():
                raise ValueError(f"Column '{col}' contains invalid characters")
        
        print("Data validation passed")

    def _basic_preprocessing(df, id_columns, target_col, date_col, ignored_features):
        """Data preprocessing operations"""
        
        # Select relevant columns
        use_cols = [c for c in df.columns 
                    if c in id_columns + [target_col, date_col] 
                    or (c not in ignored_features and c not in id_columns and c != target_col and c != date_col)]
        df = df[use_cols]
        
        # Remove zero standard deviation columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in id_columns + [target_col]]
        
        if len(numeric_cols) > 0:
            stds = df[numeric_cols].std()
            zero_std_cols = stds[stds == 0].index.tolist()
            if zero_std_cols:
                df = df.drop(columns=zero_std_cols)
                print(f"Removed {len(zero_std_cols)} zero std columns")
        
        # Fill missing values
        df = df.fillna(-999999)
        
        # Skip rounding to save memory for very wide datasets
        if len(df.columns) < 1000:
            # Round to 2 decimal places only for smaller datasets
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].round(2)
        else:
            print(f"Skipping rounding for wide dataset ({len(df.columns)} columns) to save memory")
        
        # Ensure date column is datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        return df

    def _train_test_split(df, date_col, train_start, train_end, oot_start, oot_end):
        """Train OOT temporal split"""
        
        train_mask = (df[date_col] >= train_start) & (df[date_col] <= train_end)
        oot_mask = (df[date_col] >= oot_start) & (df[date_col] <= oot_end)
        
        train_df = df[train_mask].copy()
        oot_df = df[oot_mask].copy()
        
        print(f"Train period: {train_start} to {train_end} - {len(train_df):,} samples")
        print(f"OOT period: {oot_start} to {oot_end} - {len(oot_df):,} samples")
        
        return train_df, oot_df

    def _statistical_feature_analysis(train_df, oot_df, target_col, id_columns, date_col):
        """Statistical feature selection"""
        import scipy.stats as stats
        from sklearn.feature_selection import mutual_info_classif, chi2, SelectKBest, f_classif
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score
        import warnings
        warnings.filterwarnings('ignore')
        
        feature_cols = [c for c in train_df.columns 
                       if c not in id_columns + [target_col, date_col]]
        
        if len(feature_cols) == 0:
            return []
        
        print(f"Starting enhanced statistical analysis of {len(feature_cols)} features for improved AUCPR...")
        
        # Step 0: Statistical Data Quality Pre-filtering
        print("Step 0: Statistical data quality pre-filtering...")

        def assess_feature_quality(feature_data, target_data):
            """Feature quality assessment"""
            try:
                # 1. Missing value analysis - more lenient for better features
                missing_ratio = feature_data.isnull().sum() / len(feature_data)
                if missing_ratio > 0.90:  # Increased from 0.95
                    return False, f"Excessive missing values: {missing_ratio:.2%}"
                
                # 2. For numeric features
                if pd.api.types.is_numeric_dtype(feature_data):
                    feature_clean = feature_data.dropna()
                    target_clean = target_data[~feature_data.isnull()]
                    
                    if len(feature_clean) < 10:
                        return False, "Insufficient valid values"
                    
                    # Check for zero variance
                    variance = feature_clean.var()
                    if variance == 0:
                        return False, "Zero variance (constant values)"
                    
                    # Enhanced variability check with target correlation
                    try:
                        # Quick correlation check - keep features with any correlation
                        corr = np.corrcoef(feature_clean.fillna(0), target_clean)[0, 1]
                        if not np.isnan(corr) and abs(corr) > 0.01:  # Very low threshold
                            return True, "Target correlated"
                    except:
                        pass
                    
                    # Coefficient of variation - more lenient
                    mean_val = feature_clean.mean()
                    if mean_val != 0:
                        cv = abs(variance**0.5 / mean_val)
                        if cv < 0.001:  # Reduced from 0.005
                            return False, f"Very low variability (CV: {cv:.6f})"
                
                # 3. For categorical features - enhanced assessment
                elif pd.api.types.is_object_dtype(feature_data):
                    feature_clean = feature_data.dropna()
                    target_clean = target_data[~feature_data.isnull()]
                    
                    if len(feature_clean) < 10:
                        return False, "Insufficient valid values"
                    
                    unique_count = feature_clean.nunique()
                    if unique_count <= 1:
                        return False, "Single unique value"
                    
                    # More lenient cardinality check
                    cardinality_ratio = unique_count / len(feature_clean)
                    if cardinality_ratio > 0.98:  # Increased from 0.95
                        return False, f"High cardinality: {unique_count} unique values"
                    
                    # Quick target interaction check for categorical
                    try:
                        if 2 <= unique_count <= 20:  # Reasonable categories
                            contingency = pd.crosstab(feature_clean, target_clean)
                            if contingency.shape[0] > 1 and contingency.shape[1] > 1:
                                chi2_stat, p_value = stats.chi2_contingency(contingency)[:2]
                                if p_value < 0.2:  # Lenient significance
                                    return True, "Target associated"
                    except:
                        pass
                
                return True, "Pass"
                
            except Exception as e:
                return False, f"Error in assessment: {str(e)}"
        
        # Enhanced pre-filtering with target interaction
        quality_features = []
        removed_count = 0
        
        # Combined target for assessment
        combined_target = pd.concat([train_df[target_col], oot_df[target_col]], ignore_index=True)
        
        # Process in small batches to avoid memory spikes
        batch_size = 50  # Increased batch size for efficiency
        for i in range(0, len(feature_cols), batch_size):
            batch_features = feature_cols[i:i + batch_size]
            
            for feature in batch_features:
                try:
                    # Use combined data for robust assessment
                    combined_feature_data = pd.concat([train_df[feature], oot_df[feature]], ignore_index=True)
                    is_quality, reason = assess_feature_quality(combined_feature_data, combined_target)
                    
                    if is_quality:
                        quality_features.append(feature)
                    else:
                        removed_count += 1
                        if removed_count <= 5:  # Show fewer removals
                            print(f"  Removed {feature}: {reason}")
                        elif removed_count == 6:
                            print(f"  ... (continuing quality filtering)")
                
                except Exception as e:
                    removed_count += 1
            
            # Memory cleanup after each batch
            if i % 200 == 0:  # Less frequent cleanup
                gc.collect()
        
        print(f"Statistical pre-filtering: {len(feature_cols)} -> {len(quality_features)} features")
        
        # Update feature list to quality-filtered features
        feature_cols = quality_features
        
        if len(feature_cols) == 0:
            print("No features passed quality pre-filtering")
            return []
        
        # Enhanced statistical significance testing
        if len(feature_cols) > 300:
            print(f"Still {len(feature_cols)} features - applying enhanced statistical filter...")
            
            # Multiple statistical tests for better feature selection
            significant_features = []
            y_combined = pd.concat([train_df[target_col], oot_df[target_col]], ignore_index=True)
            
            # Process more features with multiple tests
            for feature in feature_cols[:800]:  # Increased from 500
                try:
                    combined_feature = pd.concat([train_df[feature], oot_df[feature]], ignore_index=True)
                    feature_score = 0
                    
                    if pd.api.types.is_numeric_dtype(combined_feature):
                        # Multiple correlation tests
                        feature_clean = combined_feature.fillna(combined_feature.median())
                        
                        # Pearson correlation
                        try:
                            corr = np.corrcoef(feature_clean, y_combined)[0, 1]
                            if not np.isnan(corr):
                                feature_score += abs(corr) * 10
                        except:
                            pass
                        
                        # Spearman correlation for non-linear relationships
                        try:
                            spearman_corr, _ = stats.spearmanr(feature_clean, y_combined)
                            if not np.isnan(spearman_corr):
                                feature_score += abs(spearman_corr) * 8
                        except:
                            pass
                        
                        # F-statistic test
                        try:
                            f_stat, p_val = f_classif(feature_clean.values.reshape(-1, 1), y_combined)
                            if p_val[0] < 0.05:
                                feature_score += 5
                        except:
                            pass
                    
                    elif pd.api.types.is_object_dtype(combined_feature):
                        # Enhanced categorical tests
                        try:
                            contingency = pd.crosstab(combined_feature.fillna('missing'), y_combined)
                            if contingency.shape[0] > 1 and contingency.shape[1] > 1:
                                chi2, p_value, _, _ = stats.chi2_contingency(contingency)
                                if p_value < 0.05:  # More stringent
                                    feature_score += 8
                                elif p_value < 0.1:
                                    feature_score += 4
                        except:
                            pass
                        
                        # Mutual information for categorical
                        try:
                            le = LabelEncoder()
                            feature_encoded = le.fit_transform(combined_feature.fillna('missing').astype(str))
                            mi_score = mutual_info_classif(feature_encoded.reshape(-1, 1), y_combined, random_state=42)[0]
                            feature_score += mi_score * 20
                        except:
                            pass
                    
                    # Keep features with any significant score
                    if feature_score > 0.05:  # Lower threshold to keep more features
                        significant_features.append((feature, feature_score))
                
                except:
                    continue
                
                # Dynamic stopping based on quality
                if len(significant_features) >= 400:  # Increased limit
                    break
            
            # Sort by score and take top features
            significant_features.sort(key=lambda x: x[1], reverse=True)
            feature_cols = [f[0] for f in significant_features[:300]]  # Top 300
            print(f"After enhanced statistical testing: {len(feature_cols)} features")
        
        feature_scores = {}
        
        # Combine train and oot for stability analysis
        combined_df = pd.concat([train_df, oot_df], ignore_index=True)
        
        print("1. Analyzing feature stability between train and OOT...")
        
        # 1. Feature Stability Analysis (PSI-based)
        def calculate_psi(expected, actual, bins=10):
            """Population Stability Index calculation"""
            try:
                expected_clean = expected.dropna()
                actual_clean = actual.dropna()
                
                if len(expected_clean) < 10 or len(actual_clean) < 10:
                    return float('inf')  # Unstable
                
                # Create bins based on expected (train) distribution
                try:
                    bin_edges = np.quantile(expected_clean, np.linspace(0, 1, bins + 1))
                    bin_edges = np.unique(bin_edges)
                    if len(bin_edges) < 3:
                        return float('inf')
                except:
                    return float('inf')
                
                # Calculate histograms
                expected_hist, _ = np.histogram(expected_clean, bins=bin_edges)
                actual_hist, _ = np.histogram(actual_clean, bins=bin_edges)
                
                # Avoid division by zero
                expected_pct = (expected_hist + 1e-8) / (expected_hist.sum() + 1e-7)
                actual_pct = (actual_hist + 1e-8) / (actual_hist.sum() + 1e-7)
                
                # PSI calculation
                psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
                return max(0.0, min(psi, 10.0))  # Cap at 10
            except:
                return float('inf')
        
        stable_features = []
        
        # Process features in batches to manage memory
        batch_size = 50
        for i in range(0, len(feature_cols), batch_size):
            batch_features = feature_cols[i:i+batch_size]
            
            for feature in batch_features:
                try:
                    train_values = train_df[feature]
                    oot_values = oot_df[feature]
                    
                    # Check if feature has sufficient variance
                    combined_values = combined_df[feature]
                    
                    # Skip if too many nulls
                    null_ratio = combined_values.isnull().sum() / len(combined_values)
                    if null_ratio > 0.9:
                        continue
                    
                    # For numeric features
                    if combined_values.dtype in ['float64', 'float32', 'int64', 'int32']:
                        # Check variance
                        if combined_values.var() == 0:
                            continue
                        
                        # Calculate PSI with more lenient thresholds for better feature retention
                        psi = calculate_psi(train_values, oot_values)
                        if psi < 0.4:  # More lenient PSI threshold (was 0.25)
                            stable_features.append((feature, psi, 'numeric'))
                        elif psi < 1.0:  # Include moderately unstable features if they have predictive power
                            # Check if feature has strong correlation despite instability
                            try:
                                combined_feature = pd.concat([train_values, oot_values], ignore_index=True)
                                combined_target = pd.concat([train_df[target_col], oot_df[target_col]], ignore_index=True)
                                corr = np.corrcoef(combined_feature.fillna(0), combined_target)[0, 1]
                                if not np.isnan(corr) and abs(corr) > 0.05:  # Strong predictive power
                                    stable_features.append((feature, psi, 'numeric'))
                            except:
                                pass
                    
                    # For categorical features
                    elif combined_values.dtype == 'object':
                        # Check if categorical has reasonable number of categories
                        n_categories = combined_values.nunique()
                        if 2 <= n_categories <= 50:  # Reasonable range
                            # Compare distributions using Chi-square test
                            try:
                                train_counts = train_values.value_counts()
                                oot_counts = oot_values.value_counts()
                                
                                # Align categories
                                all_categories = set(train_counts.index) | set(oot_counts.index)
                                train_aligned = [train_counts.get(cat, 0) for cat in all_categories]
                                oot_aligned = [oot_counts.get(cat, 0) for cat in all_categories]
                                
                                # Chi-square test for independence - more lenient thresholds
                                if sum(train_aligned) > 0 and sum(oot_aligned) > 0:
                                    chi2_stat, p_value = stats.chisquare(oot_aligned, train_aligned)
                                    if p_value > 0.01:  # More lenient threshold (was 0.05)
                                        stable_features.append((feature, p_value, 'categorical'))
                                    else:
                                        # Even if distributions differ, keep if predictive
                                        try:
                                            le = LabelEncoder()
                                            combined_feature = pd.concat([train_values, oot_values], ignore_index=True)
                                            combined_target = pd.concat([train_df[target_col], oot_df[target_col]], ignore_index=True)
                                            feature_encoded = le.fit_transform(combined_feature.fillna('missing').astype(str))
                                            mi_score = mutual_info_classif(feature_encoded.reshape(-1, 1), combined_target, random_state=42)[0]
                                            if mi_score > 0.05:  # Strong mutual information
                                                stable_features.append((feature, p_value, 'categorical'))
                                        except:
                                            pass
                            except:
                                continue
                
                except:
                    continue
            
            # Memory cleanup after each batch
            gc.collect()
        
        print(f"Found {len(stable_features)} stable features")
        
        # 2. Enhanced Target Relationship Analysis
        print("2. Analyzing enhanced target relationships...")
        
        target_correlated_features = []
        y_train = train_df[target_col]
        y_oot = oot_df[target_col]
        
        # Process features in batches for memory efficiency
        batch_size = 100
        for i in range(0, len(stable_features), batch_size):
            batch_features = stable_features[i:i+batch_size]
            
            for feature, stability_score, feature_type in batch_features:
                try:
                    feature_scores = []
                    
                    if feature_type == 'numeric':
                        train_feature = train_df[feature].fillna(train_df[feature].median())
                        oot_feature = oot_df[feature].fillna(oot_df[feature].median())
                        
                        # Multiple correlation measures for robustness
                        # Pearson correlation
                        try:
                            corr_train = stats.pearsonr(train_feature, y_train)[0]
                            corr_oot = stats.pearsonr(oot_feature, y_oot)[0]
                            if not np.isnan(corr_train) and not np.isnan(corr_oot):
                                avg_corr = (abs(corr_train) + abs(corr_oot)) / 2
                                feature_scores.append(avg_corr)
                        except:
                            pass
                        
                        # Spearman correlation for non-linear relationships
                        try:
                            spearman_train, _ = stats.spearmanr(train_feature, y_train)
                            spearman_oot, _ = stats.spearmanr(oot_feature, y_oot)
                            if not np.isnan(spearman_train) and not np.isnan(spearman_oot):
                                avg_spearman = (abs(spearman_train) + abs(spearman_oot)) / 2
                                feature_scores.append(avg_spearman * 0.8)  # Weight slightly less
                        except:
                            pass
                        
                        # Point-biserial correlation (specific for binary targets)
                        try:
                            pb_corr, _ = stats.pointbiserialr(y_train, train_feature)
                            if not np.isnan(pb_corr):
                                feature_scores.append(abs(pb_corr))
                        except:
                            pass
                    
                    elif feature_type == 'categorical':
                        try:
                            # Enhanced categorical analysis
                            le = LabelEncoder()
                            train_encoded = le.fit_transform(train_df[feature].fillna('missing').astype(str))
                            
                            # Mutual information
                            mi_score = mutual_info_classif(train_encoded.reshape(-1, 1), y_train, random_state=42)[0]
                            feature_scores.append(mi_score * 2)  # Weight higher for MI
                            
                            # Chi-square test
                            contingency = pd.crosstab(train_df[feature].fillna('missing'), y_train)
                            if contingency.shape[0] > 1 and contingency.shape[1] > 1:
                                chi2_stat, p_value, _, _ = stats.chi2_contingency(contingency)
                                if p_value < 0.05:
                                    feature_scores.append(0.1)  # Bonus for significance
                        except:
                            continue
                    
                    # Take maximum score from all measures
                    if feature_scores:
                        max_score = max(feature_scores)
                        if max_score > 0.01:  # Reduced threshold
                            target_correlated_features.append((feature, max_score, stability_score, feature_type))
                
                except:
                    continue
            
            # Memory cleanup
            if i % 300 == 0:
                gc.collect()
        
        print(f"Found {len(target_correlated_features)} target-correlated features")
        
        # 3. Advanced Final Selection with ML-Based Scoring
        print("3. Computing advanced feature scores with ML validation...")
        
        # Enhanced combined scoring
        final_scores = []
        for feature, target_score, stability_score, feature_type in target_correlated_features:
            if feature_type == 'numeric':
                # For numeric: higher target score, lower PSI is better
                combined_score = target_score * (2 / (1 + stability_score))
            else:
                # For categorical: higher target score, higher stability p-value is better
                combined_score = target_score * min(stability_score * 2, 1.0)
            
            final_scores.append((feature, combined_score, feature_type, target_score))
        
        # Sort by combined score
        final_scores.sort(key=lambda x: x[1], reverse=True)
        
        # ML-based validation for top features
        if len(final_scores) > 50:
            print(f"Validating top features with Random Forest...")
            
            # Take top candidates for ML validation
            top_candidates = [f[0] for f in final_scores[:min(1000, len(final_scores))]]
            
            try:
                # Prepare data for ML validation
                X_train_sample = train_df[top_candidates].fillna(-999999).head(min(10000, len(train_df)))
                y_train_sample = train_df[target_col].head(min(10000, len(train_df)))
                
                # Quick Random Forest for feature importance
                rf = RandomForestClassifier(n_estimators=20, max_depth=6, random_state=42, n_jobs=1)
                rf.fit(X_train_sample, y_train_sample)
                
                # Get feature importance and combine with statistical scores
                rf_importance = dict(zip(top_candidates, rf.feature_importances_))
                
                # Re-score with ML validation
                ml_enhanced_scores = []
                for feature, stat_score, ftype, target_score in final_scores:
                    if feature in rf_importance:
                        # Combine statistical score with ML importance
                        ml_score = rf_importance[feature]
                        enhanced_score = (stat_score * 0.6) + (ml_score * 0.4)
                        ml_enhanced_scores.append((feature, enhanced_score, ftype))
                
                final_scores = [(f, score, ftype, 0) for f, score, ftype in ml_enhanced_scores]
                final_scores.sort(key=lambda x: x[1], reverse=True)
                
                print(f"ML validation completed for {len(ml_enhanced_scores)} features")
                
            except Exception as e:
                print(f"ML validation failed: {e}, using statistical scores only")
        
        # Optimized selection focused on stability and predictive power
        if len(final_scores) > 0:
            # Strategy: Keep all features that show ANY predictive power and are stable
            # Remove arbitrary limits that may hurt model performance
            
            scores_only = [score for _, score, _, _ in final_scores]
            score_mean = np.mean(scores_only)
            score_std = np.std(scores_only)
            
            # Use a very permissive threshold - keep features with any statistical significance
            # Focus on removing only clearly useless features
            min_meaningful_score = max(0.001, score_mean - 2 * score_std)  # Very permissive
            
            # Separate by stability (PSI) for different treatment
            high_stability_features = []  # PSI < 0.25 (very stable)
            moderate_stability_features = []  # PSI 0.25-0.5 (moderately stable)
            
            for feature, score, ftype, target_score in final_scores:
                if score > min_meaningful_score:  # Any meaningful predictive power
                    # Find the stability score from our earlier analysis
                    stability_info = None
                    for stable_feat, stability_score, feat_type in stable_features:
                        if stable_feat == feature:
                            stability_info = (stability_score, feat_type)
                            break
                    
                    if stability_info:
                        stability_score, feat_type = stability_info
                        if feat_type == 'numeric' and stability_score < 0.4:  # Stable (updated threshold)
                            high_stability_features.append(feature)
                        elif feat_type == 'numeric' and stability_score < 1.0:  # Moderately stable
                            moderate_stability_features.append(feature)
                        elif feat_type == 'categorical' and stability_score > 0.01:  # Categorical with reasonable stability
                            high_stability_features.append(feature)
                        elif feat_type == 'categorical':  # Include all categorical if they have predictive power
                            moderate_stability_features.append(feature)
                    else:
                        # If no stability info, include if it has strong predictive power
                        if score > score_mean:
                            moderate_stability_features.append(feature)
            
            # Build final selection prioritizing stable features
            selected_features = high_stability_features.copy()
            
            # Add moderate features if we have room and they add value
            for feat in moderate_stability_features:
                if feat not in selected_features:
                    selected_features.append(feat)
            
            # If still too few features, add more from final_scores
            if len(selected_features) < 20:  # Very minimal threshold
                for feature, score, ftype, target_score in final_scores:
                    if feature not in selected_features and score > 0:
                        selected_features.append(feature)
                        if len(selected_features) >= 50:  # Reasonable minimum
                            break
            
            # Only apply upper limit if memory is truly constraining
            # Allow up to 500 features if they're all meaningful
            if len(selected_features) > 500:
                # Sort by combined score and take top 500
                sorted_features = sorted([(f, s) for f, s, _, _ in final_scores if f in selected_features], 
                                       key=lambda x: x[1], reverse=True)
                selected_features = [f[0] for f in sorted_features[:500]]
                print(f"Reduced to top 500 features due to memory constraints")
            
            print(f"Selected {len(selected_features)} features using stability-focused selection")
            print(f"  - High stability features: {len(high_stability_features)}")
            print(f"  - Moderate stability features: {len(moderate_stability_features)}")
        else:
            selected_features = []
        
        print(f"Final selection: {len(selected_features)} features using enhanced statistical analysis")

        # Print top features with their scores
        print("Top 15 selected features:")
        for i, (feature, score, ftype, target_score) in enumerate(final_scores[:15]):
            status = "SELECTED" if feature in selected_features else "REJECTED"
            print(f"  {status} {i+1}. {feature} ({ftype}): combined={score:.6f}, target={target_score:.6f}")
        
        return selected_features
    
    def _feature_selection_lgbm_batched(train_df, target_col, id_columns, date_col, max_features=1000):
        """LightGBM feature selection fallback"""
        
        feature_cols = [c for c in train_df.columns 
                       if c not in id_columns + [target_col, date_col]]
        
        if len(feature_cols) == 0:
            return []
        
        # Conservative feature limit based on memory constraints
        max_features_for_memory = min(max_features, 1000)  # Conservative limit for memory efficiency
        
        if len(feature_cols) > max_features_for_memory:
            print(f"MEMORY CRITICAL: {len(feature_cols)} features detected, reducing to {max_features_for_memory} to prevent OOM")
            
            # Ultra-fast feature selection - only check first 500 features to avoid memory issues
            print(f"Quick filtering first 500 features to avoid memory spike...")
            valid_features = []
            
            # Process in smaller batches to avoid memory spikes
            batch_size = 100
            for i in range(0, min(500, len(feature_cols)), batch_size):
                batch_cols = feature_cols[i:i+batch_size]
                for col in batch_cols:
                    try:
                        # Quick checks without loading too much data
                        if str(train_df[col].dtype) in ['float64', 'float32', 'int64', 'int32']:
                            # Sample check to avoid full column scan
                            sample_data = train_df[col].head(1000)
                            null_ratio = sample_data.isnull().sum() / len(sample_data)
                            if null_ratio < 0.8:  # Less than 80% nulls in sample
                                var_val = sample_data.var()
                                if pd.notna(var_val) and var_val > 0:
                                    valid_features.append(col)
                                    if len(valid_features) >= max_features_for_memory:
                                        break
                    except:
                        continue
                
                if len(valid_features) >= max_features_for_memory:
                    break
                    
                # Force garbage collection after each batch
                gc.collect()
            
            print(f"Selected {len(valid_features)} features using ultra-fast selection")
            feature_cols = valid_features[:max_features_for_memory]
        
        print(f"Starting batched feature selection with {len(feature_cols)} features on {len(train_df)} samples...")
        
        # Sample data if too large to fit in memory
        sample_size = min(50000, len(train_df))
        if len(train_df) > sample_size:
            print(f"Sampling {sample_size} rows for feature selection")
            train_sample = train_df.sample(n=sample_size, random_state=42)
        else:
            train_sample = train_df
        
        X = train_sample[feature_cols].fillna(-999999)
        y = train_sample[target_col]
        
        # Use faster LightGBM settings for large datasets
        lgb_train = lgb.Dataset(X, label=y)
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 10,  # Further reduced for speed
            'learning_rate': 0.2,  # Increased for faster convergence
            'feature_fraction': 0.6,  # Reduced to handle more features
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'force_col_wise': True,
            'max_depth': 4  # Limit depth for speed
        }
        
        print("Training LightGBM for feature importance (fast mode)...")
        model = lgb.train(params, lgb_train, num_boost_round=3, callbacks=[lgb.log_evaluation(0)])
        
        # Get feature importance
        importance = model.feature_importance()
        feature_imp = list(zip(feature_cols, importance))
        feature_imp.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Feature importance calculated. Top 5: {feature_imp[:5]}")
        
        # Select top features - limit to reasonable number
        max_selected = min(1000, len(feature_cols))  # Cap at 200 features
        selected_features = [feat for feat, imp in feature_imp[:max_selected] if imp > 0]
        
        print(f"Selected {len(selected_features)} features using batched LGBM")
        
        # Clean up
        del X, y, train_sample, lgb_train, model
        gc.collect()
        
        return selected_features

    def _get_dataset_info(dataset_path):
        """Dataset metadata extraction"""
        import pyarrow.parquet as pq
        
        # Use pyarrow to get metadata without loading data
        parquet_file = pq.ParquetFile(dataset_path)
        schema = parquet_file.schema_arrow
        total_rows = parquet_file.metadata.num_rows
        
        # Get column names
        columns = [field.name for field in schema]
        
        # Get approximate dtypes by reading just a small sample
        sample_df = pd.read_parquet(dataset_path).head(100)
        dtypes = sample_df.dtypes
        
        return columns, total_rows, dtypes
    
    def _load_data_chunked(dataset_path, chunk_size=10000):
        """Memory efficient data loading"""
        columns, total_rows, dtypes = _get_dataset_info(dataset_path)
        print(f"Dataset info: {total_rows} rows, {len(columns)} columns")
        
        # For very wide datasets, use a different approach
        if len(columns) > 2000:
            print(f"Very wide dataset ({len(columns)} columns) - using memory-efficient loading")
            # Load the entire dataset but with immediate optimization
            print("Loading wide dataset with immediate memory optimization...")
            df = pd.read_parquet(dataset_path)
            print(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")
            
            # CRITICAL: Immediately optimize to reduce memory before processing
            print("Applying aggressive memory optimization...")
            df = _optimize_dtypes(df)
            
            # Force garbage collection to free up memory
            gc.collect()
            _monitor_memory()
            
            # All columns preserved for comprehensive statistical analysis
            print("All columns preserved for statistical feature analysis")
            
            return df
        
        # For normal datasets, use chunked loading
        chunks = []
        processed_rows = 0
        
        # Since pandas doesn't support chunksize for parquet, we'll load it all
        try:
            df = pd.read_parquet(dataset_path)
            df = _optimize_dtypes(df)
            print(f"Loaded and optimized dataset: {df.shape}")
            _monitor_memory()
            return df
        except MemoryError:
            print("Memory error - dataset too large to load at once")
            raise MemoryError("Dataset too large for available memory")

    print("Starting data preprocessing...")
    _monitor_memory()
    
    # Load data in chunks to avoid memory issues
    df = _load_data_chunked(dataset_input, chunk_size=5000)
    print(f"Loaded dataset: {df.shape}")
    _monitor_memory()

    # Data validation
    _validate_data(df, id_columns, target_col, date_col, train_start, train_end, oot_start, oot_end)

    # Basic preprocessing
    df = _basic_preprocessing(df, id_columns, target_col, date_col, ignored_features)
    _monitor_memory()

    # Train-test split
    train_df, oot_df = _train_test_split(df, date_col, train_start, train_end, oot_start, oot_end)
    
    # Clean up original dataframe to free memory
    del df
    gc.collect()
    _monitor_memory()

    # Comprehensive statistical feature selection
    print("Starting comprehensive statistical feature analysis...")
    try:
        selected_features = _statistical_feature_analysis(train_df, oot_df, target_col, id_columns, date_col)
    except Exception as e:
        print(f"Statistical analysis failed: {e}, falling back to LightGBM selection")
        selected_features = _feature_selection_lgbm_batched(train_df, target_col, id_columns, date_col)

    # Keep only selected features
    keep_cols = id_columns + [target_col, date_col] + selected_features
    train_df = train_df[keep_cols]
    oot_df = oot_df[keep_cols]

    # Save results
    train_df.to_parquet(train_output)
    oot_df.to_parquet(oot_output)

    # Save feature report
    report = {
        'selected_features': selected_features,
        'train_shape': train_df.shape,
        'oot_shape': oot_df.shape
    }
    with open(feature_selection_report, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Preprocessing complete - Train: {train_df.shape}, OOT: {oot_df.shape}")

    outputs = namedtuple('Outputs', ['train_shape', 'oot_shape', 'selected_features', 'status'])
    return outputs(str(train_df.shape), str(oot_df.shape), str(selected_features), "success")

