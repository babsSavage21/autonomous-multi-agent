import streamlit as st
from agentFramework import agentFramework
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

usecaseName = st.text_input("Usecase Name", "Enter the name of the use case")
businessProfile = st.text_area("Business Profile", "Enter the detailed business profile")

bisRulesCol, inputDataCol = st.columns(2)
with bisRulesCol:
    businessRules = st.text_area("Business Rules", "Enter the detailed business rules")
with inputDataCol:
    inputData = st.text_area("Input Data", "Enter the input data")

if(st.button('Initiate')):
    agen_frame = agentFramework()
    directorResp = agen_frame.director_prompt_output(businessProfile, businessRules)
    st.markdown(str(directorResp))