
import pandas as pd
import streamlit as st

# Reshape logic
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

