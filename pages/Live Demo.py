import streamlit as st

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import json
import os
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import (
    FileReadTool
)
from crewai.tools import tool
from textwrap import dedent
from ibm_watson_machine_learning.foundation_models import Model
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

model_id = "ibm/granite-3-2-8b-instruct"
project_id = '7e4954e5-30e3-47a3-a156-c8b3b84da7e9'

parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 8000,
    "stop_sequences": ["Input:"],
    "repetition_penalty": 1
}

st.set_page_config(
    page_title = 'Execution Run'
)
st.html("""
    <style>
        .stMainBlockContainer {
            max-width:70rem;
        }
    </style>
    """
)
st.title('Autonomous Multi-Agent Framework')
st.subheader("*powered by IBM watsonx*", divider="gray")

def get_credentials():
    return {
        "url" : "https://us-south.ml.cloud.ibm.com",
        "apikey" : "szAY0l-S9imIeyTUyn3zq5VlTlglzIqKLDJGnefWr6SZ"
    }

def get_llm_models(model_id, parameters, project_id):
    model = Model(
            model_id = model_id,
            params = parameters,
            credentials = get_credentials(),
            project_id = project_id
            )
    return model
custom_headers = {
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "Gecko/20100101 Firefox/135.0"
    )
}

def get_soup(url):
    response = requests.get(url, headers=custom_headers)

    if response.status_code != 200:
        print("Error in getting webpage")
        exit(-1)

    return BeautifulSoup(response.text, "lxml")
    
def extract_review(review, is_local=True):
    author = review.select_one(".a-profile-name").text.strip()
    rating = (
        review.select_one(".review-rating > span").text
        .replace("out of 5 stars", "")
        .strip()
    )
    date = review.select_one(".review-date").text.strip()
    
    if is_local:
        title = (
            review.select_one(".review-title")
            .select_one("span:not([class])")
            .text.strip()
        )
        content = ' '.join(
            review.select_one(".review-text").stripped_strings
        )
        img_selector = ".review-image-tile"
    else:
        title = (
            review.select_one(".review-title")
            .select_one("span:not([class])").title
        )
        content = ' '.join(
            review.select_one(".review-text").stripped_strings
        )
        img_selector = ".review-image-tile"
    
    verified_element = review.select_one("span.a-size-mini")
    verified = verified_element.text.strip() if verified_element else None

    image_elements = review.select(img_selector)
    images = (
        [img.attrs["data-src"] for img in image_elements] 
        if image_elements else None
    )

    return {
        "type": "local" if is_local else "global",
        "author": author,
        "rating": rating,
        "title": title,
        "content": content.replace("Read more", ""),
        "date": date,
        "verified": verified,
        "images": images
    }

def get_reviews(soup):
    reviews = []
    
    local_reviews = soup.select("#cm-cr-dp-review-list > li")
    global_reviews = soup.select("#cm-cr-global-review-list > li")
    
    for review in local_reviews:
        reviews.append(extract_review(review, is_local=True))
    
    for review in global_reviews:
        reviews.append(extract_review(review, is_local=False))
    return reviews

def main(search_url):
    review_content= []
    soup = get_soup(search_url)
    title_element = soup.select_one('#productTitle')
    title = title_element.text.strip()
    
    description = soup.select_one(
    '#productDescription, #feature-bullets > ul').text.strip()

    reviews = get_reviews(soup)
    for review in reviews:
        review_content.append(review['content'])
    final_json = {
        "productname":title,
        "productdescription":description,
        "productreviews":review_content
    }
    df = pd.DataFrame(final_json)

    return(df)

def extract_json_string(text):
    pattern = re.compile(r'```JSON(.*?)```', re.DOTALL | re.IGNORECASE)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    else:
        return text


def get_agent_prompt_input(inputData):
    prompt_input = f"""You are an autonomous AI application. Based on the company profile and task define the worker agent with its role, goal and backstory in JSON format. Generate only the JSON output. Do not provide any explanation or notes.
```JSON
{{
  "worker_agent": {{
    "name": "",
    "role": "",
    "goal": "",
    "backstory": ""
  }}
}}
```
Input: {inputData}
Output:"""
    return prompt_input

def get_task_prompt_input(inputData):
    prompt_input  = f"""Based on the worker agent and task details define the task description and expected output. The expected output should also define the format of the output which is expected. Identify whether the agent is self-sufficient to execute the task or it needs external tools. The list of 5 available tools and their description are mentioned below. Only select a tool if it is necessary to execute the task and the task cannot be done by LLM. For some tasks more than on tool might be needed to execute the task.

1. VisionTool: This tool is used to extract text from images. When passed to the agent it will extract the text from the image and then use it to generate a response, report or any other output. The URL or the PATH of the image should be passed to the Agent.
2. ClassificationTool: The ClassificationTool is designed to classify given data based on their classes.
3. FetchProductRivewTool: A custom tool designed to extract the product details like name, description and customer review comments from a specified URL of a website.
4. FileReadTool: The FileReadTool conceptually represents a suite of functionalities within the crewai_tools package aimed at facilitating file reading and content retrieval. This suite includes tools for processing batch text files, reading runtime configuration files, and importing data for analytics. It supports a variety of text-based file formats such as .txt, .csv, .json, and more. Depending on the file type, the suite offers specialized functionality, such as converting JSON content into a Python dictionary for ease of use.
5. LinkupSearchTool: Tool Description: The LinkupSearchTool provides the ability to query the Linkup API for contextual information and retrieve structured results. This tool is ideal for enriching workflows with up-to-date and reliable information from Linkup, allowing agents to access relevant data during their tasks.

Generate the response in the given JSON format. Generate only the JSON output. Do not provide any additional texts like explanation or notes.

```JSON
{{
	"task_description": "",
	"expected_output": "",
	"tool_names": ["", ""]
}}
```

Input: {inputData}
Output:"""
    return prompt_input

def master_agent_prompt_output(companyProfile, BusinessRule):
    ma_prompt_input1 = "##Company Profile##\n" +  companyProfile + "\n\n##Business Rule##\n" + BusinessRule       
    prompt_input = f"""You are an autonomous AI application. Based on the company profile and business rules identify the worker agents, task, task description including the business conditions and expected output. The output should be in below JSON format with the name, task, task description and task output for each agent. All the fields in the JSON are mandatory.

```JSON
{{
"worker_agents": [
    {{
        "name": "",
        "task": "",
        "task_description": "",
        "task_output": ""
    }},
    {{
        "name": "",
        "task": "",
        "task_description": "",
        "task_output": ""
    }},
    {{
        "name": "",
        "task": "",
        "task_description": "",
        "task_output": ""
    }}
]
}}
```

Input: {ma_prompt_input1}
Output:"""
  
    model = get_llm_models(model_id, parameters, project_id)
    ma_generated_response = model.generate_text(prompt=prompt_input)
    ma_result = (ma_generated_response+" ")[:ma_generated_response.find("Input:")]
    final_ma_config = ''
    ma_result_json = json.loads(extract_json_string(ma_result))

    for val in ma_result_json['worker_agents']:
        worker_agent_name = val['name']
        worker_agent_task = val['task']
        final_ma_config += "Worker Agent: :blue[" + worker_agent_name + "]\n\n" + "Task: :violet[" + worker_agent_task + "]\n\n\n"

    
    ma_resp_json = {
            "type":"Master Agent",
            "value":"Based on the business requirement, Master Agent has identified the below Worker Agent(s) and Task(s):" + "\n\n\n" + final_ma_config,
            "maResult": ma_result_json
            }
    final_ma_resp_json=ma_resp_json

    return final_ma_resp_json

def worker_agents_prompt_output(companyProfile, BusinessRule, ma_result):    
    final_agent_result = []
    final_agent_config = ''

    for val in ma_result["worker_agents"]: 
        agent_name = val['name']
        agent_task = val['task']
        agent_desc = val['task_description']
        agent_output = val['task_output']
        agent_prompt_input1 = "##Company Profile##\n" +  companyProfile  + "\n\n##Business Rule##\n" + BusinessRule + "\n\n##Worker Agent Name##\n"  +agent_name + "\n\n##Task Name##\n" + agent_task + "\n\n##Task Description##\n" + agent_desc + "\n\n##Task Output##\n" + agent_output
        agent_prompt_input = get_agent_prompt_input(agent_prompt_input1)

        model = get_llm_models(model_id, parameters, project_id)
        agent_generated_response = model.generate_text(prompt=agent_prompt_input)
        agent_result = extract_json_string(agent_generated_response)
        agent_result_json = json.loads(agent_result)

        agent_name = agent_result_json['worker_agent']['name']
        agent_role = agent_result_json['worker_agent']['role']
        agent_goal = agent_result_json['worker_agent']['goal']
        final_agent_config += ":blue["+agent_name.upper() + "]\n\n" + "**Role:** " + agent_role + "\n\n" + "**Goal:** " + agent_goal + "\n\n\n"

        final_agent_result.append(agent_result)

    agent_resp_json = {
            "type":"Worker Agent",
            "value":"The Master Agent initialized the following Worker Agent(s) with their roles and goals:" + "\n\n\n" + final_agent_config,
            "agentResult": final_agent_result
            }
    final_agent_resp_json=agent_resp_json

    return final_agent_resp_json

def task_prompt_output(companyProfile, ma_result, agent_result):
    
    final_task_result = []
    ma_data = ma_result['worker_agents']
    final_task_config = ''

    agent_result_data = [json.loads(agent.strip())['worker_agent'] for agent in agent_result]

    for master_agent, result_agent in zip(ma_data, agent_result_data): 
        task_name = master_agent['name']
        task_description = master_agent['task_description']
        role = result_agent['role']        
        name = result_agent['name']
        goal = result_agent['goal']
        backstory = result_agent['backstory']
        
        task_prompt_input1 = "##Company Profile##\n" +  companyProfile  + "\n\n##Worker Agent##\n" + name + "\n\n##Worker Agent Role##\n"  + role + "\n\n##Worker Agent Goal##\n" + goal + "\n\n##Worker Agent Backstory##\n" + backstory + "\n\n##Task Name##\n" + task_name + "\n\n##Task Description##\n" + task_description

        task_prompt_input = get_task_prompt_input(task_prompt_input1)
        model_id = "meta-llama/llama-3-2-90b-vision-instruct"
        model = get_llm_models(model_id, parameters, project_id)
        task_generated_response = model.generate_text(prompt=task_prompt_input)
        task_generated_response = extract_json_string(task_generated_response)
        task_result = (task_generated_response+" ")[:task_generated_response.find("Input:")]
        task_result_json = json.loads(task_result)
        task_desc = task_result_json['task_description']
        task_exp_output = task_result_json['expected_output']
        final_task_config += ":violet["+role.upper() + "]\n\n\n" + "**Description:** " + "\n\n" + task_desc + "\n\n" + "**Expected Output:** " + "\n\n" + str(task_exp_output) + "\n\n\n\n"

        final_task_result.append(task_result)
        
    task_resp_json = {
            "type":"Task",
            "value":"The Master Agent assigned the following task(s):" + "\n\n" + final_task_config,
            "taskResult": final_task_result

            }
    final_task_resp_json=task_resp_json

    return final_task_resp_json

def multi_agent_crew(max_iter, maResult, agentResult, taskResult, userQuery):     
    try:
        llm = LLM(
            model="watsonx/ibm/granite-3-2-8b-instruct",
            stop="Input:",
            max_tokens=4000,
            temperature=0,
            api_base = "https://us-south.ml.cloud.ibm.com",
            api_key= "szAY0l-S9imIeyTUyn3zq5VlTlglzIqKLDJGnefWr6SZ",
            project_id = project_id
        )
      
        agent_result = agentResult
        currentAgents = []

        for json_str in agent_result:
            agent_info = json.loads(json_str)
            currentAgents.append(Agent(
            role=agent_info['worker_agent']['role'],
            goal=agent_info['worker_agent']['goal'],
            backstory=agent_info['worker_agent']['backstory'],
            tools=[],
            llm=llm,
            max_iter=max_iter,
            verbose=True,
            ))

        task_result = taskResult
        currentTaks=[]
        file_data = ''
        for idx, currTask in enumerate(task_result):
            task_info = json.loads(currTask.strip())
            task_tool = task_info['tool_names']
            
            for tool_name in task_tool:
                if tool_name=='FileReadTool':
                    file_read_tool = FileReadTool(file_path=userQuery)
                    file_data = file_read_tool.run()

                elif tool_name=='FetchProductRivewTool':
                    file_data = main(userQuery)
                
                else:
                    file_data = userQuery            

            description = dedent(f"""
                {task_info['task_description']}
                
                Data: {file_data}
            """)
            
            agent = currentAgents[idx]
            currentTaks.append(Task(
            description=description,
            agent=agent,
            context=[],
            expected_output=str(task_info['expected_output']),
        ))


        final_crew = Crew(
        agents=currentAgents,
        tasks=currentTaks,
        verbose=True,
        full_output=True,
        output_log_file = 'logs_new.json',
        process=Process.sequential
        )

        
        final_output = final_crew.kickoff()
        
        final_response = []

        for agent_role, output_val in zip(currentAgents, final_output.tasks_output):
            answer_dict = {
                "agent":agent_role.role,
                "output":output_val.raw
            }
            final_response.append(answer_dict)
            
        crew_final_response = final_response

    except Exception as exp:
        model_execution_errors=[{"errorDetails":str(exp)}]
        modelResult=[]
        final_response_json={"modelResult":modelResult,"modelExecutionErrors":model_execution_errors}
        return final_response_json      

    return crew_final_response

def getExampleSet(setName):
    if setName == 'Smart Assistant Test Automation':
        ucName = 'Smart Assistant Test Automation'
        businessProfile= 'A software assurance provider which provides services on software testing including web applications, smart assistants, chatbots, etc. The organization specializes in automating test case scenario generation for Generative AI based smart assistants. They use Generative AI large language models to generate test cases ensuring a full coverage of the knowledge base'
        businessRules = '''1. Fetch the file from the path provided in the input. Read the content of the file and assign the extracted content as the knowledge base. Use the knowledge base to execute the below steps.

2. Generate test scenarios based on the knowledge base. Test scenarios must cover a wide range of user queries to ensure maximum coverage. Test scenarios should include variations of user queries to account for different user interaction patterns.
a) Generate only 5 questions and 3 variations of each of the 5 generated questions based on the knowledge base. The questions should be colloquial in nature imitating how users type question in chatbots.
b) Generate 3 answer variations for the each of the 5 generated questions provided in step 1(a) using the knowledge base. Assign the generated answers as ground truth which will be used for execution of the test scenario.

3. Simulate a smart assistant and execute the 5 test scenarios generated in step 1. Generate answers for each of the test scenario and assign the generated answers as smart assistant answers. The generated answers should not match the 3 answer variations generated in step 1(b) but it should be factually correct.

4. After execution of the test scenarios in step 2, do a semantic similarity of the answers generated for each test scenario with the ground truth answers generated in step 1. Based on the semantic similarity score mark the test scenarios as PASS or FAIL assuming the threshold value of 0.7.
        '''
        inputData = './test_data/content.txt'        
        return ucName, businessProfile, businessRules, inputData
    else :
        ucName = 'Product Review System'
        businessProfile= 'A new age e-commerse serving the needs of today’s smart and value seeking customers. It offers a one-stop shopping experience by offering fresh produce, bakery, dairy products, home and personal care products, general merchandise, smart apparels and appliances, making it a complete shopping destination.'
        businessRules = '''1. Analyze the reviews given by customers and classify each of the review into the following categories:
Positive
Neutral
Negative

2. From the positive reviews generate the positive review summary.

3. From the negative reviews generate the negative review summary.

4. From the customer reviews, for each review extract product features mentioned in that review and classify the sentiment expressed by the customer for that feature. For example:
Review
These are the best dairy free creamers that I have tried! Perfect amount of sweet and creamy. I can finally enjoy coffee again. Now the only issue is finding it!
Features - Sentiment
sweet and creamy (Taste) - Positive
availability (Product Availability) - Negative

5. From the customer reviews generate a summarized review.

6. Use the positive review summary from step 2 to generate the Search Engine Optimization content consisting of the following:
Product description
H1 headings
Broad Match Keywords
Meta Title
Meta Description
X (Twitter) Text
'''
        inputData = 'https://www.amazon.in/dp/B07MXXW3RL'
        return ucName, businessProfile, businessRules, inputData

selectedOption = st.selectbox(
    "Please select to load an existing example or go for a new run",
    ("Smart Assistant Test Automation", "Product Review System", "New Run"),
)

if selectedOption == 'New Run':
    usecaseName = st.text_input("Usecase Name", placeholder="Enter the name of the use case")
    businessProfile = st.text_area("Business Profile", placeholder="Enter the detailed business profile", height =68)

    bisRulesCol, inputDataCol = st.columns(2)
    with bisRulesCol:
        businessRules = st.text_area("Business Rules", placeholder="Enter the detailed business rules")
    with inputDataCol:
        inputData = st.text_area("Input Data", placeholder="Enter the input data")
else :
    ucName, ebusinessProfile, ebusinessRules, einputData = getExampleSet(selectedOption)
    
    usecaseName = st.text_input("Usecase Name",value = ucName, placeholder="Enter the name of the use case")
    businessProfile = st.text_area("Business Profile",value = ebusinessProfile, placeholder="Enter the detailed business profile", height =68)

    bisRulesCol, inputDataCol = st.columns(2)
    with bisRulesCol:
        businessRules = st.text_area("Business Rules",value = ebusinessRules, placeholder="Enter the detailed business rules", height =160)
    with inputDataCol:
        inputData = st.text_area("Input Data",value = einputData, placeholder="Enter the input data", height =160)

if(st.button('Initiate', type="secondary")):
    masterAgentResp = master_agent_prompt_output(businessProfile, businessRules)
    agentResp = worker_agents_prompt_output(businessProfile, businessRules, masterAgentResp['maResult'])
    taskResp = task_prompt_output(businessProfile, masterAgentResp['maResult'], agentResp['agentResult'])          
    finalOutput = multi_agent_crew(5, masterAgentResp['maResult'], agentResp['agentResult'], taskResp['taskResult'], inputData)         
    st.divider()
    st.header("Execution", divider="gray")

    maRespCol, agentRespCol, taskRespCol = st.columns(3)
    with maRespCol:
        with st.container():
            st.subheader("Master Agent")
        with st.container(height=250, border=True):
            st.markdown(masterAgentResp['value'])
    with agentRespCol:
        with st.container():
            st.subheader("Worker Agent")
        with st.container(height=250, border=True):
            st.markdown(agentResp['value'])
    with taskRespCol:
        with st.container():
            st.subheader("Task Details")
        with st.container(height=250, border=True):
            st.markdown(taskResp['value']) 

    st.header("Final Output", divider="gray")           

    for currAgentOp in finalOutput:
        with st.expander('Agent: ' + currAgentOp['agent']):
            if currAgentOp['output'].startswith('json'):
                viewJson = currAgentOp['output'][:len(currAgentOp['output'])]
                st.json(json.loads(viewJson))
            else :
                st.markdown(currAgentOp['output'])
