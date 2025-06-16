# Streamlit app entry point
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ML Model imports
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, classification_report, confusion_matrix, roc_curve, auc
from lifelines import CoxPHFitter, KaplanMeierFitter

# Statistical analysis
from scipy import stats
from scipy.stats import ttest_ind, chi2_contingency, f_oneway, pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Import your existing modules
from queries import (get_site_details_all, get_subjects, get_visits, get_forms, 
                    get_fields, get_lov, get_all_data)
from app import reshape_data, perform_advanced_statistics, create_advanced_visualizations, perform_ml_analysis, create_survival_analysis

# Page configuration
st.set_page_config(
    layout="wide", 
    page_title="Clinical Data Analysis Dashboard",
    page_icon="🏥",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    # .main-header {
    #     font-size: 2.5rem;
    #     font-weight: bold;
    #     color: #1f77b4;
    #     text-align: center;
    #     margin-bottom: 1rem;
    # }
    # .metric-card {
    #     background-color: #f0f2f6;
    #     padding: 1rem;
    #     border-radius: 0.5rem;
    #     border-left: 5px solid #1f77b4;
    #     margin: 0.5rem 0;
    # }
    # .analysis-section {
    #     background-color: #ffffff;
    #     padding: 1.5rem;
    #     border-radius: 0.5rem;
    #     box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    #     margin: 1rem 0;
    # }
    # .stTabs [data-baseweb="tab-list"] {
    #     gap: 2px;
    # }
    # .stTabs [data-baseweb="tab"] {
    #     height: 50px;
    #     padding-left: 20px;
    #     padding-right: 20px;
    # }
    
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
def initialize_session_state():
    """Initialize session state variables"""
    if 'raw_df' not in st.session_state:
        st.session_state.raw_df = None
    if 'wide_df' not in st.session_state:
        st.session_state.wide_df = None
    if 'analysis_ready' not in st.session_state:
        st.session_state.analysis_ready = False


# Main application
def main():
    initialize_session_state()
    
    st.markdown('<h1 class="main-header">🏥 Clinical Data Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.write("Comprehensive clinical trial data analysis with database integration")
    
    # Your existing data selection logic
    df = get_site_details_all()
    name_to_id = dict(zip(df['HospitalName'], df['SiteID']))
    
    # Show site details
    with st.expander("📊 Site Details", expanded=False):
        st.dataframe(df, use_container_width=True)
    
    # --- SIDEBAR DATA FILTERS ---
    st.sidebar.title("🎛️ Dashboard Controls")
    st.sidebar.header("Data Filters")
    
    # Site selection
    selected_name = st.sidebar.selectbox("Select Site", options=df['HospitalName'].unique())
    selected_id = name_to_id[selected_name]
    st.sidebar.success(f"Selected: {selected_name} (ID: {selected_id})")
    
    # Subject selection
    subjects = pd.DataFrame(get_subjects(selected_id))
    if not subjects.empty:
        selected_subject = st.sidebar.selectbox("Select Subject", options=subjects['SubjectName'].unique())
        
        # Visit selection
        visits = pd.DataFrame(get_visits())
        selected_visit = st.sidebar.selectbox("Select Visit", options=visits['VisitName'].unique())
        v_name_to_id = dict(zip(visits['VisitName'], visits['VisitID']))
        visit_id = v_name_to_id[selected_visit]
        
        # Form selection
        forms = pd.DataFrame(get_forms(visit_id))
        if not forms.empty:
            selected_form = st.sidebar.selectbox("Select Form", options=forms['PanelName'].unique())
            form_name_to_id = dict(zip(forms['PanelName'], forms['PanelID']))
            selected_form_id = form_name_to_id[selected_form]
            
            # Field selection
            fields = pd.DataFrame(get_fields(str(visit_id), str(selected_form_id)))
            if not fields.empty:
                selected_fields = st.sidebar.multiselect("Select Fields", options=fields['AttributeName'].unique())
                
                if selected_fields:
                    field_name_to_id = dict(zip(fields['AttributeName'], fields['DyanamicAttributeID']))
                    selected_field_ids = [field_name_to_id[x] for x in selected_fields]
                    selected_field_id_str = ", ".join(selected_field_ids)
                    
                    # Fetch data button
                    if st.sidebar.button("🔄 Fetch Data", type="primary"):
                        with st.spinner("Fetching data from database..."):
                            raw_df = pd.DataFrame(get_all_data(selected_field_id_str))
                            st.session_state.raw_df = raw_df
                            st.sidebar.success("✅ Data fetched successfully!")
    
    # --- DATA RESHAPING SECTION ---
    if st.session_state.raw_df is not None and not st.session_state.raw_df.empty:
        raw_df = st.session_state.raw_df
        
        st.header("🔄 Data Reshaping Configuration")
        
        with st.expander("Configure Data Reshaping", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Data Preview")
                st.dataframe(raw_df.head(), use_container_width=True)
            
            with col2:
                st.subheader("Reshaping Settings")
                all_cols = raw_df.columns.tolist()
                
                # Heuristic defaults
                id_defaults = [col for col in ['SubjectID'] if col in all_cols]
                param_default_options = [col for col in ['Day-2 (PARA)'] if col in all_cols]
                param_default = param_default_options[0] if param_default_options else (all_cols[0] if all_cols else None)
                value_default_options = [col for col in ['Day-2 (RES)'] if col in all_cols]
                value_default = value_default_options[0] if value_default_options else (all_cols[-1] if all_cols else None)
                
                id_vars = st.multiselect(
                    "🔑 Identifier Columns", 
                    options=all_cols, 
                    default=id_defaults,
                    help="Columns that identify unique records (e.g., SubjectID, SiteID)"
                )
                
                columns_col = st.selectbox(
                    "📋 Parameter Name Column", 
                    options=all_cols,
                    index=all_cols.index(param_default) if param_default in all_cols else 0,
                    help="Column containing parameter names (e.g., 'Pulse Rate', 'Blood Pressure')"
                )
                
                values_col = st.selectbox(
                    "📊 Value Column", 
                    options=all_cols,
                    index=all_cols.index(value_default) if value_default in all_cols else 0,
                    help="Column containing the actual measurement values"
                )
                
                if st.button("🔄 Reshape & Analyze Data", type="primary"):
                    with st.spinner("Reshaping data..."):
                        wide_df = reshape_data(raw_df, id_vars, columns_col, values_col)
                        if wide_df is not None:
                            st.session_state.wide_df = wide_df
                            st.session_state.analysis_ready = True
                            st.success("✅ Data successfully reshaped!")
                        else:
                            st.error("❌ Failed to reshape data. Please check your column selections.")
    
    # --- MAIN ANALYSIS SECTION ---
    if st.session_state.analysis_ready and st.session_state.wide_df is not None:
        wide_df = st.session_state.wide_df
        raw_df = st.session_state.raw_df
        
        # Get column types
        numeric_columns = wide_df.select_dtypes(include=np.number).columns.tolist()
        categorical_columns = wide_df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove SiteID from numeric analysis if it exists
        if 'SiteID' in numeric_columns:
            numeric_columns.remove('SiteID')
        
        st.success(f"🎉 Analysis Ready! Found {len(numeric_columns)} numeric and {len(categorical_columns)} categorical columns.")
        
        # Create comprehensive analysis tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "📋 Data Overview",
            "📊 Descriptive Analytics", 
            "📈 Advanced Visualizations",
            "🔬 Statistical Testing",
            "🤖 Machine Learning",
            "📉 Regression Analysis",
            "⏱️ Survival Analysis",
            "🎯 Clinical Insights"
        ])
        
        # --- TAB 1: DATA OVERVIEW ---
        with tab1:
            st.header("📋 Data Overview")
            
            # Basic metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", len(wide_df))
            with col2:
                st.metric("Numeric Columns", len(numeric_columns))
            with col3:
                st.metric("Categorical Columns", len(categorical_columns))
            with col4:
                st.metric("Missing Values", wide_df.isnull().sum().sum())
            
            # Data preview
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Reshaped (Wide) Data")
                st.dataframe(wide_df, use_container_width=True)
            
            with col2:
                st.subheader("Data Quality Summary")
                quality_df = pd.DataFrame({
                    'Column': wide_df.columns,
                    'Data Type': [str(dtype) for dtype in wide_df.dtypes],
                    'Missing Count': wide_df.isnull().sum().values,
                    'Missing %': (wide_df.isnull().sum().values / len(wide_df) * 100).round(2),
                    'Unique Values': [wide_df[col].nunique() for col in wide_df.columns]
                })
                st.dataframe(quality_df, use_container_width=True)
            
            with st.expander("🔍 Original Raw Data"):
                st.dataframe(raw_df, use_container_width=True)
        
        # --- TAB 2: DESCRIPTIVE ANALYTICS ---
        with tab2:
            st.header("📊 Descriptive Analytics")
            
            if numeric_columns:
                st.subheader("📈 Numeric Variables Summary")
                desc_stats = wide_df[numeric_columns].describe()
                st.dataframe(desc_stats, use_container_width=True)
                
                # Distribution plots
                st.subheader("📊 Distribution Analysis")
                selected_numeric = st.selectbox("Select variable for distribution analysis:", numeric_columns)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    wide_df[selected_numeric].hist(bins=20, ax=ax, alpha=0.7, color='skyblue')
                    ax.set_title(f'Distribution of {selected_numeric}')
                    ax.set_xlabel(selected_numeric)
                    ax.set_ylabel('Frequency')
                    st.pyplot(fig)
                
                with col2:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    wide_df.boxplot(column=selected_numeric, ax=ax)
                    ax.set_title(f'Box Plot of {selected_numeric}')
                    st.pyplot(fig)
            
            if categorical_columns:
                st.subheader("📋 Categorical Variables Summary")
                for col in categorical_columns[:3]:  # Show first 3 categorical columns
                    st.write(f"**{col}**")
                    value_counts = wide_df[col].value_counts()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.dataframe(value_counts.head(10))
                    with col2:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        value_counts.head(10).plot(kind='bar', ax=ax, color='lightcoral')
                        ax.set_title(f'Distribution of {col}')
                        ax.set_xlabel(col)
                        ax.set_ylabel('Count')
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
        
        # --- TAB 3: ADVANCED VISUALIZATIONS ---
        with tab3:
            st.header("📈 Advanced Visualizations")
            
            # Visualization controls
            col1, col2, col3 = st.columns(3)
            
            with col1:
                viz_type = st.selectbox(
                    "Select visualization type:", 
            options=[
                "Enhanced Scatter Plot", "Distribution Comparison", "Box Plot Analysis", 
                "Violin Plot", "Correlation Matrix", "Parallel Coordinates", "3D Scatter"
            ])
            
            with col2:
                x_var = st.selectbox("X-axis variable:", options=numeric_columns + categorical_columns)
            
            with col3:
                y_var_options = [None] + numeric_columns + categorical_columns
                y_var = st.selectbox("Y-axis variable (optional):", options=y_var_options)
            
            col1b, col2b, col3b = st.columns(3)
            with col1b:
                color_var_options = [None] + categorical_columns + numeric_columns
                color_var = st.selectbox("Color variable (optional):", options=color_var_options)
            with col2b:
                facet_var_options = [None] + categorical_columns
                facet_var = st.selectbox("Facet variable (optional):", options=facet_var_options)

            # Generate and display the plot
            if st.button("📊 Generate Plot", type="primary"):
                with st.spinner("Creating visualization..."):
                    fig = create_advanced_visualizations(wide_df, viz_type, x_var, y_var, color_var, facet_var)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Could not generate the plot. Check your variable selections.")
        
        # --- TAB 4: STATISTICAL TESTING ---
        with tab4:
            st.header("🔬 Statistical Testing")
            
            test_type = st.selectbox(
                "Select statistical test:",
                options=["T-Test (Independent)", "ANOVA", "Correlation", "Chi-Square"]
            )
            
            col1, col2, col3 = st.columns(3)
            var1, var2, group_var = None, None, None
            
            if test_type == "T-Test (Independent)":
                with col1:
                    var1 = st.selectbox("Select Numeric Variable:", options=numeric_columns)
                with col2:
                    group_var = st.selectbox("Select Grouping Variable (Categorical):", options=categorical_columns)
            
            elif test_type == "ANOVA":
                with col1:
                    var1 = st.selectbox("Select Numeric Variable:", options=numeric_columns)
                with col2:
                    group_var = st.selectbox("Select Grouping Variable (Categorical):", options=categorical_columns)
            
            elif test_type == "Correlation":
                with col1:
                    var1 = st.selectbox("Select Numeric Variable 1:", options=numeric_columns)
                with col2:
                    var2 = st.selectbox("Select Numeric Variable 2:", options=numeric_columns)
            
            elif test_type == "Chi-Square":
                with col1:
                    var1 = st.selectbox("Select Categorical Variable 1:", options=categorical_columns)
                with col2:
                    var2 = st.selectbox("Select Categorical Variable 2:", options=categorical_columns)
            
            if st.button("🧪 Run Test", type="primary"):
                with st.spinner("Performing statistical analysis..."):
                    results = perform_advanced_statistics(wide_df, test_type, var1, var2, group_var)
                    
                    if results:
                        st.subheader(f"Results for {results['test_type']}")
                        
                        p_value = results.get('p_value') or results.get('pearson_p_value')
                        if p_value is not None:
                            st.metric("P-value", f"{p_value:.4f}")
                            if p_value < 0.05:
                                st.success("Statistically significant result (p < 0.05)")
                            else:
                                st.warning("Result not statistically significant (p >= 0.05)")
                        
                        st.json(results)
                    else:
                        st.error("Analysis failed. Please check your inputs.")
        
        # --- TAB 5: MACHINE LEARNING (Classification & Clustering) ---
        with tab5:
            st.header("🤖 Machine Learning: Classification & Clustering")
            
            ml_type = st.selectbox(
                "Select ML Analysis Type",
                ["Logistic Regression", "Clustering"],
                key="ml_type_select"
            )
            
            if ml_type == "Logistic Regression":
                st.subheader("Logistic Regression Setup")
                col1, col2 = st.columns([1, 2])
                with col1:
                    target_var = st.selectbox("🎯 Select Target Variable (Categorical)", options=categorical_columns)
                with col2:
                    feature_vars = st.multiselect("⚙️ Select Feature Variables (Numeric)", options=numeric_columns, default=numeric_columns[:3])
                
                if st.button("🚀 Train Logistic Regression Model", type="primary"):
                    if target_var and feature_vars:
                        with st.spinner("Training model..."):
                            results = perform_ml_analysis(wide_df, "Logistic Regression", target_var, feature_vars)
                            if "error" not in results:
                                st.success("Model trained successfully!")
                                st.metric("Model Accuracy", f"{results['accuracy']:.2%}")
                                
                                st.subheader("Classification Report")
                                report_df = pd.DataFrame(results['classification_report']).transpose()
                                st.dataframe(report_df)
                                
                                st.subheader("Feature Importance")
                                importance_df = pd.DataFrame(results['feature_importance'].items(), columns=['Feature', 'Importance']).sort_values('Importance', ascending=False)
                                st.bar_chart(importance_df.set_index('Feature'))
                                
                            else:
                                st.error(results["error"])
                    else:
                        st.warning("Please select a target and at least one feature.")
            
            elif ml_type == "Clustering":
                st.subheader("K-Means Clustering Setup")
                feature_vars = st.multiselect("⚙️ Select Features for Clustering (Numeric)", options=numeric_columns, default=numeric_columns[:2])
                
                if st.button("🔍 Find Clusters", type="primary"):
                    if len(feature_vars) >= 2:
                        with st.spinner("Performing clustering..."):
                            results = perform_ml_analysis(wide_df, "Clustering", None, feature_vars)
                            if "error" not in results:
                                st.success(f"Clustering complete! Found {results['n_clusters']} clusters.")
                                wide_df['Cluster'] = results['cluster_labels'].astype(str)
                                
                                st.subheader("Cluster Visualization")
                                fig = px.scatter(
                                    wide_df,
                                    x=feature_vars[0],
                                    y=feature_vars[1],
                                    color='Cluster',
                                    title="Cluster Plot"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.error(results['error'])
                    else:
                        st.warning("Please select at least two features for clustering.")

        # --- TAB 6: REGRESSION ANALYSIS ---
        with tab6:
            st.header("📉 Regression Analysis")
            
            reg_type = st.selectbox("Select Regression Model", ["Linear Regression", "Random Forest Regression"])
            
            col1, col2 = st.columns([1, 2])
            with col1:
                target_var = st.selectbox("🎯 Select Target Variable (Numeric)", options=numeric_columns)
            with col2:
                feature_vars = st.multiselect("⚙️ Select Feature Variables (Numeric)", options=[c for c in numeric_columns if c != target_var], default=[c for c in numeric_columns if c != target_var][:3])

            if st.button("🚀 Train Regression Model", type="primary"):
                if target_var and feature_vars:
                    with st.spinner("Training model..."):
                        results = perform_ml_analysis(wide_df, reg_type, target_var, feature_vars)
                        if "error" not in results:
                            st.success(f"{results['model_type']} trained successfully!")
                            st.metric("R² Score", f"{results['r2_score']:.4f}")
                            
                            st.subheader("Feature Importance")
                            importance_df = pd.DataFrame(results['feature_importance'].items(), columns=['Feature', 'Importance']).sort_values('Importance', ascending=False)
                            st.bar_chart(importance_df.set_index('Feature'))
                            
                            st.subheader("Predictions vs. Actuals")
                            pred_df = pd.DataFrame({'Actual': results['actual'], 'Predicted': results['predictions']})
                            fig = px.scatter(pred_df, x='Actual', y='Predicted', title='Actual vs. Predicted Values', trendline='ols')
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error(results['error'])
                else:
                    st.warning("Please select a target and at least one feature.")

        # --- TAB 7: SURVIVAL ANALYSIS ---
        with tab7:
            st.header("⏱️ Survival Analysis")
            st.write("Analyze time-to-event data using Kaplan-Meier and Cox Proportional Hazards models.")
            
            # Ensure time and event columns are numeric
            wide_df_numeric = wide_df.apply(pd.to_numeric, errors='coerce')
            
            col1, col2, col3 = st.columns(3)
            with col1:
                duration_col = st.selectbox("Select Time/Duration Column", options=numeric_columns)
            with col2:
                event_col = st.selectbox("Select Event Column (1=event, 0=censor)", options=numeric_columns)
            with col3:
                group_col = st.selectbox("Select Grouping Variable (Optional)", options=[None] + categorical_columns)
            
            if st.button("📊 Run Survival Analysis", type="primary"):
                if duration_col and event_col:
                    with st.spinner("Performing survival analysis..."):
                        results = create_survival_analysis(wide_df, duration_col, event_col, group_col)
                        if "error" not in results:
                            st.success("Survival analysis complete.")
                            
                            if 'kaplan_meier_plot' in results:
                                st.subheader("Kaplan-Meier Survival Curve")
                                st.pyplot(results['kaplan_meier_plot'])
                            
                            if 'log_rank_test' in results:
                                st.subheader("Log-Rank Test Results (for 2 groups)")
                                st.metric("P-value", f"{results['log_rank_test']['p_value']:.4f}")
                            
                            if 'cox_model' in results:
                                st.subheader("Cox Proportional Hazards Model Summary")
                                st.metric("Concordance Index", f"{results['cox_model']['concordance']:.4f}")
                                st.dataframe(results['cox_model']['summary'])
                        else:
                            st.error(results['error'])
                else:
                    st.warning("Please select duration and event columns.")

        # --- TAB 8: CLINICAL INSIGHTS ---
        with tab8:
            st.header("🎯 Clinical Insights Summary")
            st.write("This section provides a high-level summary of the clinical data.")
            
            st.info("""
            **Key Observations:**

            - **Data Profile:** The dataset contains records for **{} subjects** across **{} numeric** and **{} categorical** parameters.
            - **Data Quality:** The overall missing data percentage is **{:.2f}%**. Key variables like '{}' have the most missing values.
            - **Distributions:** The numeric variable '{}' shows a mean of **{:.2f}** and a standard deviation of **{:.2f}**.
            - **Correlations:** A quick look at the correlation matrix might reveal strong relationships between certain biomarkers.
            
            *This is an automated summary. Deeper investigation in each tab is recommended for comprehensive insights.*
            """.format(
                wide_df['SubjectID'].nunique() if 'SubjectID' in wide_df.columns else 'N/A',
                len(numeric_columns),
                len(categorical_columns),
                wide_df.isnull().sum().sum() / wide_df.size * 100 if wide_df.size > 0 else 0,
                wide_df.isnull().sum().idxmax() if not wide_df.empty else 'N/A',
                numeric_columns[0] if numeric_columns else 'N/A',
                wide_df[numeric_columns[0]].mean() if numeric_columns else 0,
                wide_df[numeric_columns[0]].std() if numeric_columns else 0
            ))
            
            st.subheader("Further Exploration Suggestions")
            st.markdown("""
            1.  **Check for Outliers:** Use the box plots in the 'Descriptive Analytics' tab to identify potential outliers in key measurements.
            2.  **Group Comparisons:** Use the 'Statistical Testing' tab to perform T-Tests or ANOVAs to see if there are significant differences between patient groups (e.g., treatment vs. placebo).
            3.  **Predictive Modeling:** Explore the 'Machine Learning' tabs to build models that can predict clinical outcomes based on baseline characteristics.
            4.  **Time-to-Event:** If your study involves follow-up over time, the 'Survival Analysis' tab is crucial for understanding event probabilities.
            """)

if __name__ == '__main__':
    main()