# pyodbc connection and run_query function

import os
import pyodbc
from dotenv import load_dotenv
import streamlit as st
import re
import pandas as pd
import numpy as np
load_dotenv()

def get_connection_for_study(study_name: str):
   try: 
        print("Connecting to the database...")
        conn_str = (
            f"Driver={os.getenv('DB_DRIVER')};"
            f"Server={os.getenv('DB_SERVER')};"
            f"Database={study_name};"        
            "Trusted_Connection=yes;"            
        )
        
        connection = pyodbc.connect(conn_str)
        if connection:
            print("connected to the database.")
        connection.add_output_converter(-155, lambda value: value.decode('latin1', errors='ignore') if value else None)
    
        return connection
   except pyodbc.Error as e:
        print("Error connecting to the database:", e)
        raise e
    
  
            

def run_query(sp: str,  params: list = [] ):
    if params is None:
        params = []
    study_name = "DataEntry"
    print(f"Study Name: {study_name}")
    print(f"Running stored procedure: {sp} with params: {params}")
    
    conn = None # Initialize conn to None
    cursor = None # Initialize cursor to None
    try:
        conn = get_connection_for_study(study_name)
        cursor = conn.cursor()
        sql = f"EXEC {sp}"
        if params:          
            placeholders = ",".join(["?"] * len(params))          
            sql = f"EXEC {sp} {placeholders}"
            print(f"SQL Statement: {sql}")
            cursor.execute(sql, params)
        else:            
            print(f"SQL Statement: {sql}")
            cursor.execute(sql)

        # Advance to first result set with data
        while cursor.description is None:
            if not cursor.nextset():
                break
        result = []

        if cursor.description:
            columns = [column[0] for column in cursor.description]
            result = [dict(zip(columns, row)) for row in cursor.fetchall()]
        print(f"Number of rows returned: {len(result)}")
        return result
    except pyodbc.ProgrammingError as pe:
        print(f"PyODBC ProgrammingError: {pe}")
        if "TVP's rows must be Sequence objects" in str(pe):
            print("\n--- DEBUG HINT ---")
            print("This specific error means a stored procedure parameter is being interpreted as a Table-Valued Parameter (TVP).")
            print("If your SQL SP is `VARCHAR(MAX)` but this error occurs, it's an unusual driver/pyodbc quirk.")
            print("If your SQL SP truly expects a TVP, ensure the corresponding Python parameter is a list of tuples (e.g., `[(value,), (value2,)]`).")
            print(f"Current parameters: {params}")
            print("--- END HINT ---\n")
        # Optionally re-raise or handle more specifically
        raise pe
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise e # Re-raise other exceptions
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()



def guess_column(columns, keyword_list):
    """
    Returns the first column name that matches any keyword in keyword_list.
    """
    for keyword in keyword_list:
        for col in columns:
            if re.search(keyword, col, re.IGNORECASE):
                return col
    return None

def describe_column_values(df, cols, max_val_len=30):
    preview = {}
    for col in cols:
        try:
            sample_val = df[col].dropna().astype(str).unique()[:1]
            preview_text = sample_val[0][:max_val_len] if len(sample_val) else "No data"
        except Exception:
            preview_text = "Invalid data"
        preview[f"{col} → e.g. '{preview_text}'"] = col
    return preview

# def reshape_data(raw_df):
    # """
    # Universally reshapes clinical trial data from long to wide format.
    # Automatically detects column names like Subject ID, Parameter, and Result using fuzzy logic.
    # """
    # if raw_df is None or raw_df.empty:
    #     st.warning("The input data is empty.")
    #     return None

    # try:
    #     all_cols = raw_df.columns.tolist()

    #     # --- Heuristic Matching for ID Columns ---
    #     id_keywords = ['subject', 'patient', 'site', 'center', 'id']
    #     id_cols = [col for col in all_cols if any(re.search(k, col, re.IGNORECASE) for k in id_keywords)]
    #     id_defaults = id_cols[:2]  # Prefer Subject ID + Site ID or similar

    #     # --- Heuristic Matching for Parameter and Value Columns ---
    #     param_col = guess_column(all_cols, ['parameter', 'param', 'test', 'measure'])
    #     value_col = guess_column(all_cols, ['result', 'value', 'res', 'reading'])

    #     # --- Fallback to Manual Selection if needed ---
    #     if not id_defaults or not param_col or not value_col:
    #         st.warning("Automatic detection failed. Please select columns manually.")
            
    #     # Columns that should not be in param/value selection
    #     excluded_cols = ['SiteID', 'SubjectID', 'Sr No.']

    #     # Filter options for parameter and value column selection
    #     param_options = [col for col in all_cols if col not in excluded_cols]
    #     param_options = [col for col in all_cols if col not in excluded_cols]
        
    #      # Filter SiteID out of ID defaults if needed
    #     id_defaults = [col for col in id_defaults if col.lower() != 'siteid']
    #     id_defaults = st.multiselect("Select ID columns", all_cols, default=id_defaults)
       
    #     # Generate display-to-actual-column mappings
    #     param_choices = describe_column_values(raw_df, param_options)
    #     value_choices = describe_column_values(raw_df, param_options)

    #     # param_col = st.selectbox(
    #     #     "Select Parameter column",
    #         # param_options,
    #         # index=param_options.index(param_col) if param_col in param_options else 0
    #     param_label = st.selectbox("Select Parameter column", list(param_choices.keys()))
    #     param_col = param_choices[param_label]
    #     # )

    #     # value_col = st.selectbox(
    #     #     "Select Value column",
    #         # value_options,
    #         # index=value_options.index(value_col) if value_col in value_options else 0
    #     value_label = st.selectbox("Select Value column", list(value_choices.keys()))
    #     value_col = value_choices[value_label]
    #     # )

       

    #     # --- Clean ID Columns ---
    #     raw_df = raw_df.dropna(subset=id_defaults)
    #     for col in id_defaults:
    #         if raw_df[col].ndim != 1:
    #             raw_df[col] = raw_df[col].apply(lambda x: x[0] if isinstance(x, (list, tuple)) else x)

    #     # --- Pivot the Data ---
    #     wide_df = raw_df.pivot_table(
    #         index=id_defaults,
    #         columns=param_col,
    #         values=value_col,
    #         aggfunc='first'  # Safe for both numeric and string values
    #     ).reset_index()

    #     wide_df.columns.name = None  # Clean up column index name
    #     st.success(f"Data reshaped successfully. Shape: {wide_df.shape}")
    #     return wide_df

    # except Exception as e:
    #     st.error(f"Error during reshaping: {e}")
    #     return None
    
# @st.cache_data(show_spinner="Reshaping data for analysis...")
def reshape_data(raw_df, id_vars, columns_col, values_col):
    """
    Reshapes data from long to wide format for analysis based on user-selected columns.
    """
    if raw_df is None or raw_df.empty:
        return None
    
    if not id_vars or not columns_col or not values_col:
        st.warning("Please select all required columns for reshaping in the sidebar.")
        return None
       
    try:
        # --- FIX: Convert the value column to a numeric type before pivoting ---
        # The 'errors='coerce'' argument will turn any non-numeric values into NaN (Not a Number)
        # which prevents the pivot operation from failing on text data.
        
        raw_df[values_col] = pd.to_numeric(raw_df[values_col], errors='coerce')
        
        # Pivot the table using the user-specified columns
        wide_df = raw_df.pivot_table(
            index=id_vars,
            columns=columns_col,
            values=values_col,
            aggfunc='mean'
        ).reset_index()
        wide_df.columns.name = None # Clean up column index name
        # st.info(f"Reshaped data to shape: {wide_df.shape}")
        # st.write("Missing values summary:")
        # st.dataframe(wide_df.isnull().sum())
        return wide_df
    except Exception as e:
        st.error(f"An error occurred while reshaping the data: {e}. Check if the selected columns are appropriate for pivoting.")
        return None

