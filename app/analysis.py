# Statistical, correlation, regression analysis functions
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
# ML Model imports
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, cross_val_score


from scipy.stats import ttest_ind, chi2_contingency, f_oneway, pearsonr, spearmanr
def perform_advanced_statistics(df, test_type, var1, var2=None, group_var=None):
    """Perform various statistical tests"""
    results = {}
    
    try:
        if test_type == "T-Test (Independent)":
            if group_var and var1 in df.columns and group_var in df.columns:
                groups = df[group_var].unique()
                if len(groups) >= 2:
                    group1_data = df[df[group_var] == groups[0]][var1].dropna()
                    group2_data = df[df[group_var] == groups[1]][var1].dropna()
                    stat, p_value = ttest_ind(group1_data, group2_data)
                    results = {
                        'test_statistic': stat,
                        'p_value': p_value,
                        'group1_mean': group1_data.mean(),
                        'group2_mean': group2_data.mean(),
                        'group1_std': group1_data.std(),
                        'group2_std': group2_data.std(),
                        'test_type': 'Independent T-Test'
                    }
        
        elif test_type == "ANOVA":
            if group_var and var1 in df.columns and group_var in df.columns:
                groups = [group[var1].dropna() for name, group in df.groupby(group_var)]
                if len(groups) >= 2:
                    stat, p_value = f_oneway(*groups)
                    results = {
                        'f_statistic': stat,
                        'p_value': p_value,
                        'test_type': 'One-way ANOVA'
                    }
        
        elif test_type == "Correlation":
            if var1 in df.columns and var2 in df.columns:
                # Remove NaN values
                clean_data = df[[var1, var2]].dropna()
                if len(clean_data) > 2:
                    pearson_corr, pearson_p = pearsonr(clean_data[var1], clean_data[var2])
                    spearman_corr, spearman_p = spearmanr(clean_data[var1], clean_data[var2])
                    results = {
                        'pearson_correlation': pearson_corr,
                        'pearson_p_value': pearson_p,
                        'spearman_correlation': spearman_corr,
                        'spearman_p_value': spearman_p,
                        'test_type': 'Correlation Analysis'
                    }
        
        elif test_type == "Chi-Square":
            if var1 in df.columns and var2 in df.columns:
                contingency_table = pd.crosstab(df[var1], df[var2])
                chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                results = {
                    'chi2_statistic': chi2,
                    'p_value': p_value,
                    'degrees_of_freedom': dof,
                    'test_type': 'Chi-Square Test'
                }
    
    except Exception as e:
        st.error(f"Error performing statistical test: {str(e)}")
        return None
    
    return results

def create_advanced_visualizations(df, viz_type, x_var, y_var=None, color_var=None, facet_var=None):
    """Create advanced visualizations using Plotly"""
    
    try:
        if viz_type == "Enhanced Scatter Plot":
            fig = px.scatter(
                df, x=x_var, y=y_var, color=color_var, facet_col=facet_var,
                title=f"{y_var} vs {x_var}",
                hover_data=df.select_dtypes(include=[np.number]).columns.tolist()[:3],
                trendline="ols" if pd.api.types.is_numeric_dtype(df[x_var]) and pd.api.types.is_numeric_dtype(df[y_var]) else None
            )
            return fig
        
        elif viz_type == "Distribution Comparison":
            fig = px.histogram(
                df, x=x_var, color=color_var, facet_col=facet_var,
                title=f"Distribution of {x_var}",
                marginal="box"
            )
            return fig
        
        elif viz_type == "Box Plot Analysis":
            fig = px.box(
                df, x=color_var, y=x_var, facet_col=facet_var,
                title=f"{x_var} by {color_var}",
                points="outliers"
            )
            return fig
        
        elif viz_type == "Violin Plot":
            fig = px.violin(
                df, x=color_var, y=x_var, facet_col=facet_var,
                title=f"{x_var} Distribution by {color_var}",
                box=True
            )
            return fig
        
        elif viz_type == "Correlation Matrix":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                fig = px.imshow(
                    corr_matrix,
                    title="Correlation Matrix",
                    color_continuous_scale="RdBu_r",
                    aspect="auto"
                )
                return fig
        
        elif viz_type == "Parallel Coordinates":
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:6]  # Limit to 6 columns
            if len(numeric_cols) > 2:
                fig = px.parallel_coordinates(
                    df, dimensions=numeric_cols.tolist(),
                    color=color_var if color_var in df.columns else numeric_cols[0],
                    title="Parallel Coordinates Plot"
                )
                return fig
        
        elif viz_type == "3D Scatter":
            if len(df.select_dtypes(include=[np.number]).columns) >= 3:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                fig = go.Figure(data=[go.Scatter3d(
                    x=df[numeric_cols[0]],
                    y=df[numeric_cols[1]],
                    z=df[numeric_cols[2]],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=df[color_var] if color_var in df.columns else df[numeric_cols[0]],
                        colorscale='Viridis',
                        showscale=True
                    )
                )])
                fig.update_layout(
                    title="3D Scatter Plot",
                    scene=dict(
                        xaxis_title=numeric_cols[0],
                        yaxis_title=numeric_cols[1],
                        zaxis_title=numeric_cols[2]
                    )
                )
                return fig
        
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return None

def perform_ml_analysis(df, analysis_type, target_var, feature_vars, test_size=0.2):
    """Perform machine learning analysis"""
    
    try:
        # Prepare data
        X = df[feature_vars].dropna()
        y = df.loc[X.index, target_var]
        
        # Remove any remaining NaN values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        if len(X) < 10:
            return {"error": "Insufficient data for ML analysis"}
        
        results = {}
        
        if analysis_type == "Linear Regression":
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            
            results = {
                'model_type': 'Linear Regression',
                'r2_score': r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'coefficients': dict(zip(feature_vars, model.coef_)),
                'intercept': model.intercept_,
                'feature_importance': dict(zip(feature_vars, np.abs(model.coef_))),
                'predictions': y_pred,
                'actual': y_test
            }
        
        elif analysis_type == "Random Forest Regression":
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            
            results = {
                'model_type': 'Random Forest Regression',
                'r2_score': r2,
                'feature_importance': dict(zip(feature_vars, model.feature_importances_)),
                'predictions': y_pred,
                'actual': y_test
            }
        
        elif analysis_type == "Logistic Regression":
            # Encode target variable if it's categorical
            if not pd.api.types.is_numeric_dtype(y):
                le = LabelEncoder()
                y_encoded = le.fit_transform(y.astype(str))
                target_classes = le.classes_
            else:
                y_encoded = y
                target_classes = np.unique(y)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_train_scaled, y_train)
            
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
            
            results = {
                'model_type': 'Logistic Regression',
                'accuracy': model.score(X_test_scaled, y_test),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'feature_importance': dict(zip(feature_vars, np.abs(model.coef_[0]))),
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'actual': y_test,
                'classes': target_classes
            }
        
        elif analysis_type == "Clustering":
            # Scale features for clustering
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Determine optimal number of clusters (up to 8)
            max_k = min(8, len(X) // 3)
            if max_k >= 2:
                inertias = []
                k_range = range(2, max_k + 1)
                
                for k in k_range:
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    kmeans.fit(X_scaled)
                    inertias.append(kmeans.inertia_)
                
                # Use elbow method or default to 3 clusters
                optimal_k = 3 if max_k >= 3 else 2
                
                model = KMeans(n_clusters=optimal_k, random_state=42)
                clusters = model.fit_predict(X_scaled)
                
                results = {
                    'model_type': 'K-Means Clustering',
                    'n_clusters': optimal_k,
                    'cluster_labels': clusters,
                    'cluster_centers': model.cluster_centers_,
                    'inertia': model.inertia_,
                    'silhouette_score': None  # Could add this
                }
            else:
                return {"error": "Insufficient data for clustering analysis"}
        
        return results
        
    except Exception as e:
        return {"error": f"ML analysis failed: {str(e)}"}

def create_survival_analysis(df, duration_col, event_col, group_col=None):
    """Perform survival analysis using lifelines"""
    
    try:
        # Prepare data
        survival_data = df[[duration_col, event_col]].copy()
        if group_col:
            survival_data[group_col] = df[group_col]
        
        # Remove NaN values
        survival_data = survival_data.dropna()
        
        if len(survival_data) < 5:
            return {"error": "Insufficient data for survival analysis"}
        
        results = {}
        
        # Kaplan-Meier Estimator
        kmf = KaplanMeierFitter()
        
        if group_col and group_col in survival_data.columns:
            # Survival analysis by groups
            groups = survival_data[group_col].unique()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for group in groups:
                group_data = survival_data[survival_data[group_col] == group]
                kmf.fit(group_data[duration_col], group_data[event_col], label=f'{group_col}={group}')
                kmf.plot_survival_function(ax=ax)
            
            ax.set_title('Kaplan-Meier Survival Curves by Group')
            ax.set_xlabel('Time')
            ax.set_ylabel('Survival Probability')
            
            # Log-rank test
            from lifelines.statistics import logrank_test
            if len(groups) == 2:
                group1_data = survival_data[survival_data[group_col] == groups[0]]
                group2_data = survival_data[survival_data[group_col] == groups[1]]
                
                log_rank_result = logrank_test(
                    group1_data[duration_col], group2_data[duration_col],
                    group1_data[event_col], group2_data[event_col]
                )
                
                results['log_rank_test'] = {
                    'test_statistic': log_rank_result.test_statistic,
                    'p_value': log_rank_result.p_value
                }
        else:
            # Overall survival curve
            kmf.fit(survival_data[duration_col], survival_data[event_col])
            
            fig, ax = plt.subplots(figsize=(10, 6))
            kmf.plot_survival_function(ax=ax)
            ax.set_title('Kaplan-Meier Survival Curve')
            ax.set_xlabel('Time')
            ax.set_ylabel('Survival Probability')
        
        results['kaplan_meier_plot'] = fig
        results['median_survival'] = kmf.median_survival_time_
        
        # Cox Proportional Hazards Model (if we have covariates)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        covariate_cols = [col for col in numeric_cols if col not in [duration_col, event_col]]
        
        if len(covariate_cols) > 0:
            cox_data = df[[duration_col, event_col] + covariate_cols[:3]].dropna()  # Limit to 3 covariates
            
            if len(cox_data) >= 10:
                cph = CoxPHFitter()
                cph.fit(cox_data, duration_col=duration_col, event_col=event_col)
                
                results['cox_model'] = {
                    'summary': cph.summary,
                    'concordance': cph.concordance_index_
                }
        
        return results
        
    except Exception as e:
        return {"error": f"Survival analysis failed: {str(e)}"}

def analyze_adverse_events(df):
    """Analyze adverse events data"""
    results = {}
    
    # Identify adverse event columns
    ae_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in 
                    ['adverse', 'ae', 'side effect', 'reaction', 'toxicity', 'safety'])]
    
    if not ae_columns:
        return {"error": "No adverse event columns identified"}
    
    try:
        # Overall AE summary
        ae_data = df[ae_columns].copy()
        
        # Count total AEs per subject
        ae_counts = ae_data.notna().sum(axis=1)
        results['ae_per_subject'] = ae_counts
        
        # AE frequency analysis
        ae_summary = {}
        for col in ae_columns:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                ae_summary[col] = df[col].value_counts().to_dict()
        
        results['ae_frequency'] = ae_summary
        
        # Severity analysis (if severity columns exist)
        severity_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in 
                         ['severity', 'grade', 'serious', 'ctcae'])]
        
        if severity_cols:
            severity_analysis = {}
            for col in severity_cols:
                severity_analysis[col] = df[col].value_counts().to_dict()
            results['severity_analysis'] = severity_analysis
        
        # System Organ Class analysis (if SOC columns exist)
        soc_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in 
                      ['system', 'organ', 'class', 'soc', 'meddra'])]
        
        if soc_cols:
            soc_analysis = {}
            for col in soc_cols:
                soc_analysis[col] = df[col].value_counts().to_dict()
            results['soc_analysis'] = soc_analysis
        
        return results
        
    except Exception as e:
        return {"error": f"AE analysis failed: {str(e)}"}


def analyze_vas_scores(df):
    """Analyze Visual Analog Scale (VAS) scores"""
    results = {}
    
    # Identify VAS columns
    vas_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in 
                     ['vas', 'visual analog', 'pain score', 'likert', 'scale', 'rating'])]
    
    if not vas_columns:
        return {"error": "No VAS/Scale columns identified"}
    
    try:
        for col in vas_columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                vas_data = df[col].dropna()
                
                results[col] = {
                    'mean': vas_data.mean(),
                    'median': vas_data.median(),
                    'std': vas_data.std(),
                    'min': vas_data.min(),
                    'max': vas_data.max(),
                    'q25': vas_data.quantile(0.25),
                    'q75': vas_data.quantile(0.75),
                    'distribution': vas_data.value_counts().to_dict()
                }
                
                # Categorize VAS scores (assuming 0-10 scale)
                if vas_data.max() <= 10:
                    categories = pd.cut(vas_data, bins=[0, 3, 6, 10], 
                                        labels=['Low (0-3)', 'Moderate (4-6)', 'High (7-10)'])
                    results[col]['categories'] = categories.value_counts().to_dict()
        
        return results
        
    except Exception as e:
        return {"error": f"VAS analysis failed: {str(e)}"}



def analyze_demographics(df):
    """Analyze demographic data"""
    results = {}
    
    # Identify demographic columns
    demo_keywords = ['age', 'sex', 'gender', 'race', 'ethnicity', 'weight', 'height', 
                     'bmi', 'education', 'marital', 'occupation', 'income']
    
    demo_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in demo_keywords)]
    
    if not demo_columns:
        return {"error": "No demographic columns identified"}
    
    try:
        for col in demo_columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Numeric demographics (age, weight, height, BMI)
                demo_data = df[col].dropna()
                results[col] = {
                    'type': 'numeric',
                    'mean': demo_data.mean(),
                    'median': demo_data.median(),
                    'std': demo_data.std(),
                    'min': demo_data.min(),
                    'max': demo_data.max(),
                    'count': len(demo_data)
                }
                
                # Age categorization
                if 'age' in col.lower():
                    age_categories = pd.cut(demo_data, bins=[0, 18, 35, 50, 65, 100], 
                                            labels=['<18', '18-34', '35-49', '50-64', '65+'])
                    results[col]['age_categories'] = age_categories.value_counts().to_dict()
                
                # BMI categorization
                if 'bmi' in col.lower():
                    bmi_categories = pd.cut(demo_data, bins=[0, 18.5, 25, 30, 40, 100], 
                                            labels=['Underweight', 'Normal', 'Overweight', 'Obese', 'Severely Obese'])
                    results[col]['bmi_categories'] = bmi_categories.value_counts().to_dict()
            
            else:
                # Categorical demographics
                demo_data = df[col].dropna()
                results[col] = {
                    'type': 'categorical',
                    'value_counts': demo_data.value_counts().to_dict(),
                    'unique_count': demo_data.nunique(),
                    'most_common': demo_data.mode().iloc[0] if len(demo_data.mode()) > 0 else None
                }
        
        return results
        
    except Exception as e:
        return {"error": f"Demographics analysis failed: {str(e)}"}


def analyze_baseline_characteristics(df):
    """Analyze baseline characteristics"""
    results = {}
    
    # Identify baseline columns
    baseline_keywords = ['baseline', 'screening', 'day 0', 'visit 1', 'pre', 'initial']
    baseline_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in baseline_keywords)]
    
    # Also include medical history and comorbidity columns
    medical_keywords = ['history', 'comorbid', 'diagnosis', 'condition', 'disease', 'disorder']
    medical_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in medical_keywords)]
    
    all_baseline_cols = list(set(baseline_columns + medical_columns))
    
    if not all_baseline_cols:
        return {"error": "No baseline characteristic columns identified"}
    
    try:
        for col in all_baseline_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Numeric baseline characteristics
                baseline_data = df[col].dropna()
                results[col] = {
                    'type': 'numeric',
                    'mean': baseline_data.mean(),
                    'median': baseline_data.median(),
                    'std': baseline_data.std(),
                    'min': baseline_data.min(),
                    'max': baseline_data.max(),
                    'normal_range_analysis': analyze_normal_ranges(baseline_data, col)
                }
            else:
                # Categorical baseline characteristics
                baseline_data = df[col].dropna()
                results[col] = {
                    'type': 'categorical',
                    'value_counts': baseline_data.value_counts().to_dict(),
                    'prevalence': (baseline_data.value_counts() / len(baseline_data) * 100).to_dict()
                }
        
        return results
        
    except Exception as e:
        return {"error": f"Baseline analysis failed: {str(e)}"}

def analyze_normal_ranges(data, column_name):
    """Analyze if values fall within normal clinical ranges"""
    # Define normal ranges for common clinical parameters
    normal_ranges = {
        'systolic': (90, 140),
        'diastolic': (60, 90),
        'heart rate': (60, 100),
        'pulse': (60, 100),
        'temperature': (36.1, 37.2),
        'glucose': (70, 100),
        'cholesterol': (0, 200),
        'hemoglobin': (12, 16),
        'hematocrit': (36, 46),
        'platelet': (150, 450),
        'wbc': (4, 11),
        'creatinine': (0.6, 1.3)
    }
    
    # Find matching normal range
    normal_range = None
    for key, range_val in normal_ranges.items():
        if key in column_name.lower():
            normal_range = range_val
            break
    
    if normal_range:
        within_normal = ((data >= normal_range[0]) & (data <= normal_range[1])).sum()
        below_normal = (data < normal_range[0]).sum()
        above_normal = (data > normal_range[1]).sum()
        
        return {
            'normal_range': normal_range,
            'within_normal': within_normal,
            'below_normal': below_normal,
            'above_normal': above_normal,
            'percent_normal': (within_normal / len(data)) * 100
        }
    
    return None
