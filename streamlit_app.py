import streamlit as st

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import json
import os
from flask import Flask, request, send_from_directory,jsonify
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import (
    FileReadTool,
    SerperDevTool
)
from crewai.tools import tool
from textwrap import dedent
from langchain_ibm import WatsonxLLM
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
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

@tool("FetchProductReviewTool")
def fetch_product_review_tool():
    """ Description """
    def get_soup(url):
        response = requests.get(url, headers=custom_headers)

        if response.status_code != 200:
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
        
        # Get both local and global reviews using the same function.
        local_reviews = soup.select("#cm-cr-dp-review-list > li")
        global_reviews = soup.select("#cm-cr-global-review-list > li")
        
        for review in local_reviews:
            reviews.append(extract_review(review, is_local=True))
        
        for review in global_reviews:
            reviews.append(extract_review(review, is_local=False))
        return reviews

    #asin = "B098FKXT8L"
    review_content= []
    search_url = f"https://www.amazon.in/dp/{asin}/"
    soup = get_soup(search_url)
    title_element = soup.select_one('#productTitle')
    title = title_element.text.strip()
    
    rating_element = soup.select_one('#acrPopover')
    rating_text = rating_element.attrs.get('title')
    rating = rating_text.replace('out of 5 stars','')

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

@tool("FileReadTool")
def file_read_tool():
    """Description"""
    dir1=os.getcwd()
    # Instantiate tools
    file_path = os.path.join(dir1 ,'data', 'logs_new.txt')
    file_read_tool = FileReadTool(file_path=file_path)
    return file_read_tool.run()

def extract_json_string(text):
    # Define the regex pattern to match the string between ```JSON and ```
    pattern = re.compile(r'```JSON(.*?)```', re.DOTALL | re.IGNORECASE)

    # Search for the pattern in the text
    match = pattern.search(text)

    if match:
        # Extract and return the matched string
        return match.group(1).strip()
    else:
        return None


def get_agent_prompt_input(inputData):
    prompt_input = f"""You are an autonomous AI application. Based on the company profile and task define the worker agent with its role, goal and backstory in JSON format. Generate only the JSON output. Do not provide any explanation or notes.

{{
"worker_agent": {{
"name": "",
"role": "",
"goal": "",
"backstory": ""
}}
}}

Input: {inputData}
Output:"""
    return prompt_input

def get_task_prompt_input(inputData):

    prompt_input  = f"""You are an autonomous AI application. Based on the worker agent and task details define the task description, expected output and the tool(s) which will be required to execute the task. The expected output should also define the format of the output which is expected.

Generate the response in the given JSON format. Generate only the JSON output. Do not provide any additional texts like explanation or notes.

```JSON
{{
"task_description": "",
"expected_output": "",
"tool_names": ["", "", ""]
}}
```

The list of available tools and their description are mentioned below. Only select a tool if it is necessary to execute the task and the task cannot be done by LLM. For some tasks more than on tool might be needed to execute the task.

[
{{
    "tool_name": "VisionTool",
    "tool_description": "This tool is used to extract text from images. When passed to the agent it will extract the text from the image and then use it to generate a response, report or any other output. The URL or the PATH of the image should be passed to the Agent."
}},
{{
    "tool_name": "SummarizerTool",
    "tool_description": "The SummarizerTool is designed to summarize given texts."
}},
{{
    "tool_name": "FetchProductRivewTool",
    "tool_description": "A custom tool designed to extract the product details like name, description and customer review comments from a specified URL of a website."
}},
{{
"tool_name": "FileReadTool",
    "tool_description": "The FileReadTool conceptually represents a suite of functionalities within the crewai_tools package aimed at facilitating file reading and content retrieval. This suite includes tools for processing batch text files, reading runtime configuration files, and importing data for analytics. It supports a variety of text-based file formats such as .txt, .csv, .json, and more. Depending on the file type, the suite offers specialized functionality, such as converting JSON content into a Python dictionary for ease of use."
}},
{{
    "tool_name": "LinkupSearchTool",
    "tool_description": "The LinkupSearchTool provides the ability to query the Linkup API for contextual information and retrieve structured results. This tool is ideal for enriching workflows with up-to-date and reliable information from Linkup, allowing agents to access relevant data during their tasks."
}}
]


Input: {inputData}
Output:"""
    return prompt_input

def director_prompt_output(companyProfile, BusinessRule): 

    dir_prompt_input1 = "##Company Profile##\n" +  companyProfile + "\n\n##Business Rule##\n" + BusinessRule       
    prompt_input = f"""You are an autonomous AI application. Based on the company profile and business rules identify the worker agents, task, task description including the business conditions and expected output. The output should be in below JSON format with the worker agents, task description and expected output for each agent.

```json
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

Input: {dir_prompt_input1}
Output:"""
  
    model = get_llm_models(model_id, parameters, project_id)
    dir_generated_response = model.generate_text(prompt=prompt_input)
    dir_result = (dir_generated_response+" ")[:dir_generated_response.find("Input:")]
    final_dir_config = ''
    dir_result_json = json.loads(extract_json_string(dir_result))

    for val in dir_result_json['worker_agents']:
        actors_name = val['name']
        actors_task = val['task']
        final_dir_config += "Worker Agent: " + actors_name + "\n" + "Task: " + actors_task + "\n\n"

    
    dir_resp_json = {
            "type":"Master Agent",
            "value":"Based on the business requirement, Master Agent has identified the below Worker Agent(s) and Task(s):" + "\n\n" + final_dir_config,
            "dirResult": dir_result_json
            }
    final_dir_resp_json=dir_resp_json

    return final_dir_resp_json

def agents_prompt_output(companyProfile, BusinessRule, dir_result):
    
    #dir_result = json.loads(dir_result)
    final_agent_result = []
    final_agent_config = ''

    for val in dir_result["worker_agents"]: 
        agent_name = val['name']
        agent_task = val['task']
        agent_desc = val['task_description']
        agent_output = val['task_output']
        agent_prompt_input1 = "##Company Profile##\n" +  companyProfile  + "\n\n##Business Rule##\n" + BusinessRule + "\n\n##Worker Agent Name##\n"  +agent_name + "\n\n##Task Name##\n" + agent_task + "\n\n##Task Description##\n" + agent_desc + "\n\n##Task Output##\n" + agent_output
        agent_prompt_input = get_agent_prompt_input(agent_prompt_input1)

        model = get_llm_models(model_id, parameters, project_id)
        agent_generated_response = model.generate_text(prompt=agent_prompt_input)
        agent_result = (agent_generated_response+" ")[:agent_generated_response.find("Input:")]
        agent_result_json = json.loads(agent_result)

        agent_name = agent_result_json['worker_agent']['name']
        agent_role = agent_result_json['worker_agent']['role']
        agent_goal = agent_result_json['worker_agent']['goal']
        final_agent_config += agent_name.upper() + "\n" + "Role: " + agent_role + "\n" + "Goal: " + agent_goal + "\n\n"


        final_agent_result.append(agent_result)

    agent_resp_json = {
            "type":"Worker Agent",
            "value":"The Master Agent initialized the following Actor(s) with their roles and goals:" + "\n\n" + final_agent_config,
            "agentResult": final_agent_result
            }
    final_agent_resp_json=agent_resp_json

    return final_agent_resp_json

def task_prompt_output(companyProfile, dir_result, agent_result):
    
    final_task_result = []
    #dir_result = json.loads(dir_result)
    dir_data = dir_result['worker_agents']
    final_task_config = ''

    agent_result_data = [json.loads(agent.strip())['worker_agent'] for agent in agent_result]

    for dir_agent, result_agent in zip(dir_data, agent_result_data): 
        task_name = dir_agent['name']
        task_description = dir_agent['task_description']
        role = result_agent['role']        
        name = result_agent['name']
        goal = result_agent['goal']
        backstory = result_agent['backstory']
        
        task_prompt_input1 = "##Company Profile##\n" +  companyProfile  + "\n\n##Worker Agent##\n" + name + "\n\n##Worker Agent Role##\n"  + role + "\n\n##Worker Agent Goal##\n" + goal + "\n\n##Worker Agent Backstory##\n" + backstory + "\n\n##Task Name##\n" + task_name + "\n\n##Task Description##\n" + task_description

        task_prompt_input = get_task_prompt_input(task_prompt_input1)
        # task_prompt_input = task_prompt_input + "\nInput: \n"+ task_prompt_input1 +"\nOutput:"
        # model_id = "meta-llama/llama-3-3-70b-instruct"
        model = get_llm_models(model_id, parameters, project_id)
        task_generated_response = model.generate_text(prompt=task_prompt_input)
        task_generated_response = extract_json_string(task_generated_response)
        task_result = (task_generated_response+" ")[:task_generated_response.find("Input:")]
        task_result_json = json.loads(task_result)

        task_desc = task_result_json['task_description']
        task_exp_output = task_result_json['expected_output']
        final_task_config += role.upper() + "\n" + "Description: " + "\n" + task_desc + "\n\n" + "Expected Output: " + "\n" + str(task_exp_output) + "\n\n\n"

        final_task_result.append(task_result)
        
    task_resp_json = {
            "type":"actor_task",
            "value":"The Master Agent assigned the following task(s):" + "\n\n" + final_task_config,
            "taskResult": final_task_result

            }
    final_task_resp_json=task_resp_json

    return final_task_resp_json

def multi_agent_crew(max_iter, dirResult, agentResult, taskResult, userQuery):     
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
      
        agent_result = agentResult#agents_prompt_output(companyProfile, dir_result, llm)
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

        task_result = taskResult  #task_prompt_output(companyProfile, dir_result, agent_result)
        currentTaks=[]

        for idx, currTask in enumerate(task_result):
            task_info = json.loads(currTask.strip())

            description = dedent(f"""
                {task_info['task_description']}
                
                Data: {userQuery}
            """)
            agent = currentAgents[idx]
            currentTaks.append(Task(
            description=description,
            agent=agent,
            tool = [fetch_product_review_tool, file_read_tool],
            context=[],
            expected_output=str(task_info['expected_output']),
        ))


        final_crew = Crew(
        agents=currentAgents,
        tasks=currentTaks,
        verbose=True,
        full_output=True,
        # output_log_file=True,
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
    if setName == 'Example1':
        ucName = ''
        businessProfile= ''
        businessRules = ''
        inputData = ''
        directorResp = {'type': 'Master Agent', 'value': 'Based on the business requirement, Master Agent has identified the below Worker Agent(s) and Task(s):\n\nWorker Agent: Knowledge Base Extraction Agent\nTask: Extract Knowledge Base\n\nWorker Agent: Test Scenario Generation Agent\nTask: Generate Test Scenarios\n\nWorker Agent: Smart Assistant Simulation Agent\nTask: Simulate Smart Assistant\n\nWorker Agent: Semantic Similarity Agent\nTask: Evaluate Test Scenarios\n\n', 'dirResult': {'worker_agents': [{'name': 'Knowledge Base Extraction Agent', 'task': 'Extract Knowledge Base', 'task_description': 'Fetch the file from the path provided in the input. Read the content of the file and assign the extracted content as the knowledge base.', 'task_output': 'The extracted content of the file as the knowledge base.'}, {'name': 'Test Scenario Generation Agent', 'task': 'Generate Test Scenarios', 'task_description': 'Generate 5 questions and 3 variations of each of the 5 generated questions based on the knowledge base. The questions should be colloquial in nature imitating how users type question in chatbots. Generate 3 answer variations for each of the 5 generated questions using the knowledge base.', 'task_output': '5 questions with 3 variations each and 3 answer variations for each of the 5 questions.'}, {'name': 'Smart Assistant Simulation Agent', 'task': 'Simulate Smart Assistant', 'task_description': 'Simulate a smart assistant and execute the 5 test scenarios generated in step 1. Generate answers for each of the test scenario and assign the generated answers as smart assistant answers. The generated answers should not match the 3 answer variations generated in step 1(b) but it should be factually correct.', 'task_output': 'Answers generated for each of the 5 test scenarios.'}, {'name': 'Semantic Similarity Agent', 'task': 'Evaluate Test Scenarios', 'task_description': 'After execution of the test scenarios in step 2, do a semantic similarity of the answers generated for each test scenario with the ground truth answers generated in step 1. Based on the semantic similarity score mark the test scenarios as PASS or FAIL assuming the threshold value of 0.7.', 'task_output': 'PASS or FAIL for each of the 5 test scenarios.'}]}}
        agentResp = {'type': 'Worker Agent', 'value': "The Master Agent initialized the following Actor(s) with their roles and goals:\n\nKNOWLEDGE BASE EXTRACTION AGENT\nRole: Extracts and processes the knowledge base from the provided file.\nGoal: To accurately extract the content of the file and assign it as the knowledge base for further processing.\n\nTEST SCENARIO GENERATION AGENT\nRole: AI Application Specialized in Generating Test Scenarios for Generative AI based Smart Assistants\nGoal: To generate a set of 5 questions with 3 variations each and 3 answer variations for each of the 5 questions based on the provided knowledge base, ensuring maximum coverage of user queries and variations in user interaction patterns.\n\nSMART ASSISTANT SIMULATION AGENT\nRole: Smart Assistant Simulator\nGoal: To simulate a smart assistant and generate answers for 5 test scenarios based on the knowledge base, ensuring the answers are factually correct and not matching the predefined answer variations.\n\nSEMANTIC SIMILARITY AGENT\nRole: Evaluates the semantic similarity of the smart assistant's responses to the ground truth answers and marks the test scenarios as PASS or FAIL\nGoal: To ensure the smart assistant's responses are factually correct and semantically similar to the ground truth answers\n\n", 'agentResult': ['\n{\n"worker_agent": {\n"name": "Knowledge Base Extraction Agent",\n"role": "Extracts and processes the knowledge base from the provided file.",\n"goal": "To accurately extract the content of the file and assign it as the knowledge base for further processing.",\n"backstory": "This agent is designed to handle the initial step of fetching and processing the knowledge base from a given file, ensuring that the subsequent stages of test scenario generation and execution can be carried out effectively."\n}\n}', '\n\n{\n"worker_agent": {\n"name": "Test Scenario Generation Agent",\n"role": "AI Application Specialized in Generating Test Scenarios for Generative AI based Smart Assistants",\n"goal": "To generate a set of 5 questions with 3 variations each and 3 answer variations for each of the 5 questions based on the provided knowledge base, ensuring maximum coverage of user queries and variations in user interaction patterns.",\n"backstory": "This agent is an autonomous AI application designed to specialize in generating test scenarios for Generative AI based smart assistants. It uses Generative AI large language models to generate test cases, ensuring full coverage of the knowledge base. The agent is trained to create colloquial questions imitating user queries in chatbots and generate factually correct answers that do not match the predefined answer variations."\n}\n}', '\n{\n"worker_agent": {\n"name": "Smart Assistant Simulation Agent",\n"role": "Smart Assistant Simulator",\n"goal": "To simulate a smart assistant and generate answers for 5 test scenarios based on the knowledge base, ensuring the answers are factually correct and not matching the predefined answer variations.",\n"backstory": "The Smart Assistant Simulation Agent is an autonomous AI application designed to mimic the behavior of a smart assistant. It is trained to understand and respond to a wide range of user queries related to software testing services, particularly in the domain of Generative AI-based smart assistants. The agent\'s primary goal is to ensure maximum coverage of user queries by generating colloquial test scenarios and providing accurate, non-predefined responses."\n}\n}', '\n{\n"worker_agent": {\n"name": "Semantic Similarity Agent",\n"role": "Evaluates the semantic similarity of the smart assistant\'s responses to the ground truth answers and marks the test scenarios as PASS or FAIL",\n"goal": "To ensure the smart assistant\'s responses are factually correct and semantically similar to the ground truth answers",\n"backstory": "The Semantic Similarity Agent is an autonomous AI application designed to evaluate the performance of smart assistants by comparing their responses to a set of predefined ground truth answers. It uses semantic similarity algorithms to determine if the assistant\'s answers are factually correct and contextually relevant. The agent was developed to ensure the high quality and accuracy of the software testing process for Generative AI-based smart assistants."\n}\n}']}
        taskResp = {'type': 'actor_task', 'value': 'The Master Agent assigned the following task(s):\n\nEXTRACTS AND PROCESSES THE KNOWLEDGE BASE FROM THE PROVIDED FILE.\nDescription: \nFetch the file from the path provided in the input. Read the content of the file and assign the extracted content as the knowledge base.\n\nExpected Output: \nThe extracted content of the file.\n\n\nAI APPLICATION SPECIALIZED IN GENERATING TEST SCENARIOS FOR GENERATIVE AI BASED SMART ASSISTANTS\nDescription: \nGenerate 5 questions and 3 variations of each of the 5 generated questions based on the knowledge base. The questions should be colloquial in nature imitating how users type question in chatbots. Generate 3 answer variations for each of the 5 generated questions using the knowledge base.\n\nExpected Output: \nA JSON object containing 5 questions with 3 variations each and 3 answer variations for each of the 5 questions. The format should be as follows:\n{\n  "questions": [\n    {\n      "question": "question_text",\n      "variations": [\n        "variation_1",\n        "variation_2",\n        "variation_3"\n      ]\n    },\n    ...\n  ],\n  "answers": [\n    {\n      "question": "question_text",\n      "answer": "answer_text"\n    },\n    ...\n  ]\n}\n\n\nSMART ASSISTANT SIMULATOR\nDescription: \nSimulate a smart assistant and execute the 5 test scenarios generated in step 1. Generate answers for each of the test scenario and assign the generated answers as smart assistant answers. The generated answers should not match the 3 answer variations generated in step 1(b) but it should be factually correct.\n\nExpected Output: \n5 distinct answers for each test scenario, in plain text format, that are factually correct and do not match the predefined answer variations.\n\n\nEVALUATES THE SEMANTIC SIMILARITY OF THE SMART ASSISTANT\'S RESPONSES TO THE GROUND TRUTH ANSWERS AND MARKS THE TEST SCENARIOS AS PASS OR FAIL\nDescription: \nAfter execution of the test scenarios in step 2, do a semantic similarity of the answers generated for each test scenario with the ground truth answers generated in step 1. Based on the semantic similarity score mark the test scenarios as PASS or FAIL assuming the threshold value of 0.7. Output: A JSON object containing the test scenario ID, the smart assistant\'s response, the ground truth answer, the semantic similarity score, and the PASS/FAIL status.\n\nExpected Output: \n{\'test_scenario_id\': \'unique_identifier\', \'smart_assistant_response\': \'text\', \'ground_truth_answer\': \'text\', \'semantic_similarity_score\': \'decimal_value\', \'PASS/FAIL\': \'PASS\'/\'FAIL\'}\n\n\n', 'taskResult': ['{\n"task_description": "Fetch the file from the path provided in the input. Read the content of the file and assign the extracted content as the knowledge base.",\n"expected_output": "The extracted content of the file.",\n"tool_names": ["FileReadTool"]\n}', '{\n"task_description": "Generate 5 questions and 3 variations of each of the 5 generated questions based on the knowledge base. The questions should be colloquial in nature imitating how users type question in chatbots. Generate 3 answer variations for each of the 5 generated questions using the knowledge base.",\n"expected_output": "A JSON object containing 5 questions with 3 variations each and 3 answer variations for each of the 5 questions. The format should be as follows:\\n{\\n  \\"questions\\": [\\n    {\\n      \\"question\\": \\"question_text\\",\\n      \\"variations\\": [\\n        \\"variation_1\\",\\n        \\"variation_2\\",\\n        \\"variation_3\\"\\n      ]\\n    },\\n    ...\\n  ],\\n  \\"answers\\": [\\n    {\\n      \\"question\\": \\"question_text\\",\\n      \\"answer\\": \\"answer_text\\"\\n    },\\n    ...\\n  ]\\n}",\n"tool_names": ["FileReadTool"]\n}', '{\n"task_description": "Simulate a smart assistant and execute the 5 test scenarios generated in step 1. Generate answers for each of the test scenario and assign the generated answers as smart assistant answers. The generated answers should not match the 3 answer variations generated in step 1(b) but it should be factually correct.",\n"expected_output": "5 distinct answers for each test scenario, in plain text format, that are factually correct and do not match the predefined answer variations.",\n"tool_names": ["FileReadTool"]\n}', '{\n"task_description": "After execution of the test scenarios in step 2, do a semantic similarity of the answers generated for each test scenario with the ground truth answers generated in step 1. Based on the semantic similarity score mark the test scenarios as PASS or FAIL assuming the threshold value of 0.7. Output: A JSON object containing the test scenario ID, the smart assistant\'s response, the ground truth answer, the semantic similarity score, and the PASS/FAIL status.",\n"expected_output": "{\'test_scenario_id\': \'unique_identifier\', \'smart_assistant_response\': \'text\', \'ground_truth_answer\': \'text\', \'semantic_similarity_score\': \'decimal_value\', \'PASS/FAIL\': \'PASS\'/\'FAIL\'}",\n"tool_names": ["SummarizerTool", "LinkupSearchTool"]\n}']}
        finalOutput = []
        return ucName, businessProfile, businessRules, inputData, directorResp, agentResp, taskResp, finalOutput
    else :
        ucName = ''
        businessProfile= ''
        businessRules = ''
        inputData = ''
        directorResp = {'type': 'Master Agent', 'value': 'Based on the business requirement, Master Agent has identified the below Worker Agent(s) and Task(s):\n\nWorker Agent: Knowledge Base Extraction Agent\nTask: Extract Knowledge Base\n\nWorker Agent: Test Scenario Generation Agent\nTask: Generate Test Scenarios\n\nWorker Agent: Smart Assistant Simulation Agent\nTask: Simulate Smart Assistant\n\nWorker Agent: Semantic Similarity Agent\nTask: Evaluate Test Scenarios\n\n', 'dirResult': {'worker_agents': [{'name': 'Knowledge Base Extraction Agent', 'task': 'Extract Knowledge Base', 'task_description': 'Fetch the file from the path provided in the input. Read the content of the file and assign the extracted content as the knowledge base.', 'task_output': 'The extracted content of the file as the knowledge base.'}, {'name': 'Test Scenario Generation Agent', 'task': 'Generate Test Scenarios', 'task_description': 'Generate 5 questions and 3 variations of each of the 5 generated questions based on the knowledge base. The questions should be colloquial in nature imitating how users type question in chatbots. Generate 3 answer variations for each of the 5 generated questions using the knowledge base.', 'task_output': '5 questions with 3 variations each and 3 answer variations for each of the 5 questions.'}, {'name': 'Smart Assistant Simulation Agent', 'task': 'Simulate Smart Assistant', 'task_description': 'Simulate a smart assistant and execute the 5 test scenarios generated in step 1. Generate answers for each of the test scenario and assign the generated answers as smart assistant answers. The generated answers should not match the 3 answer variations generated in step 1(b) but it should be factually correct.', 'task_output': 'Answers generated for each of the 5 test scenarios.'}, {'name': 'Semantic Similarity Agent', 'task': 'Evaluate Test Scenarios', 'task_description': 'After execution of the test scenarios in step 2, do a semantic similarity of the answers generated for each test scenario with the ground truth answers generated in step 1. Based on the semantic similarity score mark the test scenarios as PASS or FAIL assuming the threshold value of 0.7.', 'task_output': 'PASS or FAIL for each of the 5 test scenarios.'}]}}
        agentResp = {'type': 'Worker Agent', 'value': "The Master Agent initialized the following Actor(s) with their roles and goals:\n\nKNOWLEDGE BASE EXTRACTION AGENT\nRole: Extracts and processes the knowledge base from the provided file.\nGoal: To accurately extract the content of the file and assign it as the knowledge base for further processing.\n\nTEST SCENARIO GENERATION AGENT\nRole: AI Application Specialized in Generating Test Scenarios for Generative AI based Smart Assistants\nGoal: To generate a set of 5 questions with 3 variations each and 3 answer variations for each of the 5 questions based on the provided knowledge base, ensuring maximum coverage of user queries and variations in user interaction patterns.\n\nSMART ASSISTANT SIMULATION AGENT\nRole: Smart Assistant Simulator\nGoal: To simulate a smart assistant and generate answers for 5 test scenarios based on the knowledge base, ensuring the answers are factually correct and not matching the predefined answer variations.\n\nSEMANTIC SIMILARITY AGENT\nRole: Evaluates the semantic similarity of the smart assistant's responses to the ground truth answers and marks the test scenarios as PASS or FAIL\nGoal: To ensure the smart assistant's responses are factually correct and semantically similar to the ground truth answers\n\n", 'agentResult': ['\n{\n"worker_agent": {\n"name": "Knowledge Base Extraction Agent",\n"role": "Extracts and processes the knowledge base from the provided file.",\n"goal": "To accurately extract the content of the file and assign it as the knowledge base for further processing.",\n"backstory": "This agent is designed to handle the initial step of fetching and processing the knowledge base from a given file, ensuring that the subsequent stages of test scenario generation and execution can be carried out effectively."\n}\n}', '\n\n{\n"worker_agent": {\n"name": "Test Scenario Generation Agent",\n"role": "AI Application Specialized in Generating Test Scenarios for Generative AI based Smart Assistants",\n"goal": "To generate a set of 5 questions with 3 variations each and 3 answer variations for each of the 5 questions based on the provided knowledge base, ensuring maximum coverage of user queries and variations in user interaction patterns.",\n"backstory": "This agent is an autonomous AI application designed to specialize in generating test scenarios for Generative AI based smart assistants. It uses Generative AI large language models to generate test cases, ensuring full coverage of the knowledge base. The agent is trained to create colloquial questions imitating user queries in chatbots and generate factually correct answers that do not match the predefined answer variations."\n}\n}', '\n{\n"worker_agent": {\n"name": "Smart Assistant Simulation Agent",\n"role": "Smart Assistant Simulator",\n"goal": "To simulate a smart assistant and generate answers for 5 test scenarios based on the knowledge base, ensuring the answers are factually correct and not matching the predefined answer variations.",\n"backstory": "The Smart Assistant Simulation Agent is an autonomous AI application designed to mimic the behavior of a smart assistant. It is trained to understand and respond to a wide range of user queries related to software testing services, particularly in the domain of Generative AI-based smart assistants. The agent\'s primary goal is to ensure maximum coverage of user queries by generating colloquial test scenarios and providing accurate, non-predefined responses."\n}\n}', '\n{\n"worker_agent": {\n"name": "Semantic Similarity Agent",\n"role": "Evaluates the semantic similarity of the smart assistant\'s responses to the ground truth answers and marks the test scenarios as PASS or FAIL",\n"goal": "To ensure the smart assistant\'s responses are factually correct and semantically similar to the ground truth answers",\n"backstory": "The Semantic Similarity Agent is an autonomous AI application designed to evaluate the performance of smart assistants by comparing their responses to a set of predefined ground truth answers. It uses semantic similarity algorithms to determine if the assistant\'s answers are factually correct and contextually relevant. The agent was developed to ensure the high quality and accuracy of the software testing process for Generative AI-based smart assistants."\n}\n}']}
        taskResp = {'type': 'actor_task', 'value': 'The Master Agent assigned the following task(s):\n\nEXTRACTS AND PROCESSES THE KNOWLEDGE BASE FROM THE PROVIDED FILE.\nDescription: \nFetch the file from the path provided in the input. Read the content of the file and assign the extracted content as the knowledge base.\n\nExpected Output: \nThe extracted content of the file.\n\n\nAI APPLICATION SPECIALIZED IN GENERATING TEST SCENARIOS FOR GENERATIVE AI BASED SMART ASSISTANTS\nDescription: \nGenerate 5 questions and 3 variations of each of the 5 generated questions based on the knowledge base. The questions should be colloquial in nature imitating how users type question in chatbots. Generate 3 answer variations for each of the 5 generated questions using the knowledge base.\n\nExpected Output: \nA JSON object containing 5 questions with 3 variations each and 3 answer variations for each of the 5 questions. The format should be as follows:\n{\n  "questions": [\n    {\n      "question": "question_text",\n      "variations": [\n        "variation_1",\n        "variation_2",\n        "variation_3"\n      ]\n    },\n    ...\n  ],\n  "answers": [\n    {\n      "question": "question_text",\n      "answer": "answer_text"\n    },\n    ...\n  ]\n}\n\n\nSMART ASSISTANT SIMULATOR\nDescription: \nSimulate a smart assistant and execute the 5 test scenarios generated in step 1. Generate answers for each of the test scenario and assign the generated answers as smart assistant answers. The generated answers should not match the 3 answer variations generated in step 1(b) but it should be factually correct.\n\nExpected Output: \n5 distinct answers for each test scenario, in plain text format, that are factually correct and do not match the predefined answer variations.\n\n\nEVALUATES THE SEMANTIC SIMILARITY OF THE SMART ASSISTANT\'S RESPONSES TO THE GROUND TRUTH ANSWERS AND MARKS THE TEST SCENARIOS AS PASS OR FAIL\nDescription: \nAfter execution of the test scenarios in step 2, do a semantic similarity of the answers generated for each test scenario with the ground truth answers generated in step 1. Based on the semantic similarity score mark the test scenarios as PASS or FAIL assuming the threshold value of 0.7. Output: A JSON object containing the test scenario ID, the smart assistant\'s response, the ground truth answer, the semantic similarity score, and the PASS/FAIL status.\n\nExpected Output: \n{\'test_scenario_id\': \'unique_identifier\', \'smart_assistant_response\': \'text\', \'ground_truth_answer\': \'text\', \'semantic_similarity_score\': \'decimal_value\', \'PASS/FAIL\': \'PASS\'/\'FAIL\'}\n\n\n', 'taskResult': ['{\n"task_description": "Fetch the file from the path provided in the input. Read the content of the file and assign the extracted content as the knowledge base.",\n"expected_output": "The extracted content of the file.",\n"tool_names": ["FileReadTool"]\n}', '{\n"task_description": "Generate 5 questions and 3 variations of each of the 5 generated questions based on the knowledge base. The questions should be colloquial in nature imitating how users type question in chatbots. Generate 3 answer variations for each of the 5 generated questions using the knowledge base.",\n"expected_output": "A JSON object containing 5 questions with 3 variations each and 3 answer variations for each of the 5 questions. The format should be as follows:\\n{\\n  \\"questions\\": [\\n    {\\n      \\"question\\": \\"question_text\\",\\n      \\"variations\\": [\\n        \\"variation_1\\",\\n        \\"variation_2\\",\\n        \\"variation_3\\"\\n      ]\\n    },\\n    ...\\n  ],\\n  \\"answers\\": [\\n    {\\n      \\"question\\": \\"question_text\\",\\n      \\"answer\\": \\"answer_text\\"\\n    },\\n    ...\\n  ]\\n}",\n"tool_names": ["FileReadTool"]\n}', '{\n"task_description": "Simulate a smart assistant and execute the 5 test scenarios generated in step 1. Generate answers for each of the test scenario and assign the generated answers as smart assistant answers. The generated answers should not match the 3 answer variations generated in step 1(b) but it should be factually correct.",\n"expected_output": "5 distinct answers for each test scenario, in plain text format, that are factually correct and do not match the predefined answer variations.",\n"tool_names": ["FileReadTool"]\n}', '{\n"task_description": "After execution of the test scenarios in step 2, do a semantic similarity of the answers generated for each test scenario with the ground truth answers generated in step 1. Based on the semantic similarity score mark the test scenarios as PASS or FAIL assuming the threshold value of 0.7. Output: A JSON object containing the test scenario ID, the smart assistant\'s response, the ground truth answer, the semantic similarity score, and the PASS/FAIL status.",\n"expected_output": "{\'test_scenario_id\': \'unique_identifier\', \'smart_assistant_response\': \'text\', \'ground_truth_answer\': \'text\', \'semantic_similarity_score\': \'decimal_value\', \'PASS/FAIL\': \'PASS\'/\'FAIL\'}",\n"tool_names": ["SummarizerTool", "LinkupSearchTool"]\n}']}
        finalOutput = []
        return ucName, businessProfile, businessRules, inputData, directorResp, agentResp, taskResp, finalOutput

@st.dialog("Please Select")
def automultiagentselect():
    example1Col, example2Col, newRunCol = st.columns(3)
    with example1Col:
        if st.button("Example1", type="primary"):
            st.session_state.automultiagentselect = {"selected": "Example1"}
            st.rerun()
    with example2Col:
        if st.button("Example2", type="primary"):
            st.session_state.automultiagentselect = {"selected": "Example2"}
            st.rerun()
    with newRunCol:
        if st.button("New Run", type="primary"):
            st.session_state.automultiagentselect = {"selected": "New Run"}
            st.rerun()

if "automultiagentselect" not in st.session_state:
    automultiagentselect()
else :
    selectedOption = st.session_state.automultiagentselect['selected']
    if selectedOption == 'New Run':
        usecaseName = st.text_input("Usecase Name", placeholder="Enter the name of the use case")
        businessProfile = st.text_area("Business Profile", placeholder="Enter the detailed business profile", height =68)

        bisRulesCol, inputDataCol = st.columns(2)
        with bisRulesCol:
            businessRules = st.text_area("Business Rules", placeholder="Enter the detailed business rules")
        with inputDataCol:
            inputData = st.text_area("Input Data", placeholder="Enter the input data")

        if(st.button('Initiate', type="secondary")):
            directorResp = director_prompt_output(businessProfile, businessRules)
            agentResp = agents_prompt_output(businessProfile, businessRules, directorResp['dirResult'])
            taskResp = task_prompt_output(businessProfile, directorResp['dirResult'], agentResp['agentResult'])          
            finalOutput = multi_agent_crew(5, directorResp['dirResult'], agentResp['agentResult'], taskResp['taskResult'], inputData)         

            st.divider()
            st.header("Execution", divider="gray")

            dirRespCol, agentRespCol, taskRespCol = st.columns(3)
            with dirRespCol:
                with st.container():
                    st.subheader("**Master Agent**")
                with st.container(height=250, border=True):
                    st.markdown(directorResp['value'])
            with agentRespCol:
                with st.container():
                    st.subheader("**Worker Agent**")
                with st.container(height=250, border=True):
                    st.markdown(agentResp['value'])
            with taskRespCol:
                with st.container():
                    st.subheader("**Task Details**")
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
    else :
        ucName, ebusinessProfile, ebusinessRules, einputData, directorResp, agentResp, taskResp, finalOutput = getExampleSet(selectedOption)
        
        usecaseName = st.text_input("Usecase Name",value = ucName, placeholder="Enter the name of the use case")
        businessProfile = st.text_area("Business Profile",value = ebusinessProfile, placeholder="Enter the detailed business profile", height =68)

        bisRulesCol, inputDataCol = st.columns(2)
        with bisRulesCol:
            businessRules = st.text_area("Business Rules",value = ebusinessRules, placeholder="Enter the detailed business rules")
        with inputDataCol:
            inputData = st.text_area("Input Data",value = einputData, placeholder="Enter the input data")
        
        st.divider()
        st.header("Execution", divider="gray")

        dirRespCol, agentRespCol, taskRespCol = st.columns(3)
        with dirRespCol:
            with st.container():
                st.subheader("**Master Agent**")
            with st.container(height=250, border=True):
                st.markdown(directorResp['value'])
        with agentRespCol:
            with st.container():
                st.subheader("**Worker Agent**")
            with st.container(height=250, border=True):
                st.markdown(agentResp['value'])
        with taskRespCol:
            with st.container():
                st.subheader("**Task Details**")
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
