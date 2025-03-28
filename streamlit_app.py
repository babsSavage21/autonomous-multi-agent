import streamlit as st

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from agentFramework import agentFramework

usecaseName = st.text_input("Usecase Name", placeholder="Enter the name of the use case")
businessProfile = st.text_area("Business Profile", placeholder="Enter the detailed business profile")

bisRulesCol, inputDataCol = st.columns(2)
with bisRulesCol:
    businessRules = st.text_area("Business Rules", placeholder="Enter the detailed business rules")
with inputDataCol:
    inputData = st.text_area("Input Data", placeholder="Enter the input data")

if(st.button('Initiate')):
    agen_frame = agentFramework()
    directorResp = agen_frame.director_prompt_output(businessProfile, businessRules)
    agentResp = agen_frame.agents_prompt_output(businessProfile, businessRules, directorResp)
    taskResp = agen_frame.task_prompt_output(businessProfile, directorResp, agentResp)
    finalOutput = agen_frame.multi_agent_crew(5, taskResp)

    dirRespCol, agentRespCol, taskRespCol = st.columns(3)
    with dirRespCol:
        st.markdown(str(directorResp))
    with agentRespCol:
        st.markdown(str(agentResp))
    with taskRespCol:
        st.markdown(str(taskResp)) 

    st.markdown(str(finalOutput))
