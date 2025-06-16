# Functions like get_site_details_all()
from db import run_query
import pandas as pd
import streamlit as st
def get_site_details_all():
    data = run_query("usp_GetSiteName_ALL",)
    df=pd.DataFrame(data)    
    return df

def get_subjects(id):
    data = run_query("usp_GetDistinct_SubjectId_ForW2GridSearch", [id])
    df=pd.DataFrame(data)    
    return df

def get_visits():
    data = run_query("usp_VisitDetails_SelectAll")
    df = pd.DataFrame(data)
    return df

def get_forms(visit_ids):
    data = run_query("usp_SelectFormFor_MultiVisitIDS", [visit_ids],)
    df = pd.DataFrame(data)
    return df


def get_fields( visit_form_ids_str, form_ids_str ):
 
    visit_ids = []
    if visit_form_ids_str:
        try:
            visit_ids = [int(x.strip()) for x in visit_form_ids_str.split(',')]
        except ValueError:
            #raise HTTPException(status_code=400, detail="Invalid visit form IDs")
            print("Warning: Invalid visit_form_ids format.")
            return {"data": [], "message": "Invalid visit_form_ids format."}, 400
    form_ids = []
    if form_ids_str:
        try:
            form_ids = [int(x.strip()) for x in form_ids_str.split(',')]
        except ValueError:
            print("Warning: Invalid form_ids format.")
            print("Warning: Invalid form_ids format.")
            return {"data": [], "message": "Invalid form_ids format."}, 400
        print(f"Parsed Visit Form IDs: {visit_ids}")
        print(f"Parsed Form IDs: {form_ids}")
   
    sp_visit_ids = ','.join(map(str, visit_ids))
    sp_form_ids = ','.join(map(str, form_ids))
    db_params = [sp_visit_ids, sp_form_ids]
    print(db_params)
    data = run_query("usp_SelectFieldIDName_ForMultipleVisitForm", db_params,)
    df = pd.DataFrame(data)
    return df


def get_lov(attributeId):
    data = run_query("usp_LOVValues_ForDataAnalyst", [attributeId])
    df = pd.DataFrame(data)
    return df
      
# @st.cache_data(show_spinner="Fetching data...")
def get_all_data( uid):
    data = run_query("usp_GetData_For_Analysis", [uid])
    st.write(data)
    df = pd.DataFrame(data)
    return df