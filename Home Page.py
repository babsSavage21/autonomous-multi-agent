import streamlit as st

st.set_page_config(
    page_title = 'Homepage'
)
st.html("""
    <style>
        .stMainBlockContainer {
            max-width:90rem;
        }

        ol {
            padding-left: 40px;
        }
    </style>
    """
)


st.title('Autonomous Multi-Agent Framework')
st.subheader("*powered by IBM watsonx*", divider="gray")

set1info = '''
<h3>Problem Statement: Limitations of Non-Autonomous Agentic Framework</h3>

<div>Agentic frameworks have greatly complemented Generative AI by enhancing its creativity, problem-solving capabilities, adaptability and efficiency. However, a non-autonomous agentic framework lacks the ability to make independent decisions or take actions without explicit instructions. Following are some of the major limitations.</div>
<ol>
    <li><b>Lack of Flexibility:</b> As they follow pre-defined rules and instructions, hence they work within a boundary and lack the ability to initiate actions on their own.</li>
    <li><b>Dependency on External Input:</b> Performance of these agents is affected if the user’s input is unclear, incorrect, incomplete or delayed.</li>
    <li><b>Limited Creativity:</b> These agents might struggle with tasks that require creative problem-solving as they operate within the constraints of pre-defined rules and instructions.</li>
    <li><b>Limited Personalization:</b> Without autonomy, these agents may struggle to provide highly personalized experiences. They can't actively gather and utilize user-specific information to tailor their responses.</li>
    <li><b>Limited Generalization:</b> These agents may struggle to generalize knowledge from one domain to another, as they lack the autonomy to explore and learn from new environments.</li>
</ol>
Several multi-agent frameworks are available in the industry like CrewAI, LangGraph and IBM’s BeeAI. These frameworks provide different architectures for implementing multi-agent based applications, but none of them provides a fully autonomous agent.

<h3>Solution: Autonomous Multi-Agent Framework powered by IBM Watsonx</h3>

<div>This integrated framework leverages IBM watsonx's Generative AI capabilities to enable dynamic creation of autonomous worker agents based on business profiles and business rules. In our approach, a sophisticated multi-agent framework, emerges as a singular, autonomous Master Agent that orchestrates the creation of worker agents and tasks at runtime to achieve the business objectives. Our approach uses CrewAI framework as the base to implement our framework. The same approach can be extended to other frameworks.</div>

'''
set2info = '''
**Key Components**

    1.<b>Master Agent Development:</b> The Master Agent analyses and extracts objectives from user-provided business requirements and rules, decomposes those objectives into steps and creates an execution plan.
    2.<b>Agent Creation:</b> The Master Agent dynamically creates worker agents at runtime based on the decomposed steps, generating details such as role, goal, and backstory for each worker agent.
    3.<b>Task Generation:</b> The Master Agent analyses the specifics of the assignment given to each worker agent, based on which it allocates tasks with descriptions and the expected output.
    4.<b>Tool Calling:</b> The Master agent evaluates if the worker agents are self-sufficient to execute the task or they require external assistance. If deemed required, relevant tools are automatically selected from the Toolkit Library and instantiated and assigned to the agents for execution.
    5.<b>Collaboration:</b> Through callback mechanism the Master Agent monitors the execution of the worker agents and enables sharing of information between worker agents ensuring successful execution of the entire operation.

**Embracing the Future with Autonomous Agents**

This autonomous multi-agent framework can boost enterprise productivity and efficiency by streamlining processes, freeing up human workers for strategic tasks, and reducing costs. It can also enhance decision-making by analysing patterns and trends, and scale with your enterprise as it grows and adapts to changing demands.

**Role of watsonx.ai**

We have leveraged multiple capabilities and features of IBM watsonx.ai to build this autonomous multi-agent framework. Here is a detailed overview of the key components and integrations:
    1.<b>Prompt Lab:</b> Custom prompts are crafted in the Prompt Lab using the "granite-3-2-8b-instruct" model. This enables the dynamic creation of autonomous worker agents, tasks, and identification of tools based on the business profile and business rules provided as input by the user. The Prompt Lab facilitates the generation of highly tailored and contextually relevant prompts, ensuring that the agents are well-equipped to handle specific business needs.
    2.<b>IBM watsonx.ai and CrewAI Integration:</b> The multi-agent framework, comprising dynamic agents, tasks, in-built tools, and custom tools, is built, orchestrated, and executed using CrewAI in integration with IBM watsonx.ai's "granite-3-2-8b-instruct" model. This model supports enhanced reasoning capabilities, making it ideal for creating autonomous agents with configurable thinking capabilities. This means the reasoning can be controlled and applied as needed, ensuring optimal performance and adaptability.
    3.<b>Prompt Executing & LLM Inferencing:</b> The watsonx.ai Runtime service offered by IBM watsonx.ai is used to execute the designed prompts as well as the crew. This service ensures seamless execution and integration, allowing the agents to perform their tasks efficiently and effectively.
    4.<b>watsonx.ai Studio:</b> IBM watsonx.ai provides a studio where we can design and implement our solutions. We have extensively used Jupyter notebooks to develop our application code and test the outcomes before deployment.
By leveraging these advanced capabilities, we have created a robust and adaptable multi-agent framework that can dynamically respond to various business needs, ensuring enhanced efficiency and effectiveness in task execution.

**Issues Faced during Hackathon**

    1.<b>Usage Limit:</b> During our implementation we reached 80% of our usage limit, due to which we could not experiment with other features like refinement of planning, usage of long-term / short-term memory, tool calling issues, etc. Also we could not test the framework for other business problems.
    2.<b>Prompts as a Service:</b> IBM watsonx.ai also provisions the deployment of finalized prompts using the WML service. However, due to the constraints of the watsonx challenge environment, we were unable to create deployments. Therefore, the prompts as a service feature could not be fully utilized in this particular environment.
    3.<b>Agent Governance:</b> We wanted to try out with Agent governance using watsonx.governance, as governance of autonomous agents is of utmost importance to ensure that Agents are consistent and predictable, maintain quality and relevance and are legal and ethical.

**Live Demo**

We have provided implementation of two different use cases using the Autonomous Multi-Agent framework. The details of the two use cases are:

    1.<b>Smart Assistant Test Automation:</b> This use case automates the testing of smart assistants.
        a. It reads the folder to establish the knowledge base using the FileReadTool of CrewAI
        b. Generates questions which can be asked by users to the smart assistant.
        c. Since the same question can be asked in a different way by different users, hence it creates alternative questions.
        d. To assist the test manager to establish the ground truth, it generates alternative answers.
        e. It mimics a smart assistant and generates answers for the generated questions.
        f. Finally, it uses semantic similarity matching to evaluate the smart assistants’ answers with the ground truth to generate the test result.
    
    2.<b>Product Review System:</b> This use case performs analytics on customer reviews.
        a. Fetches the product reviews from the provided link using a custom tool
        b. Classifies the review as Positive, Negative or Neutral
        c. Uses the positive reviews to generate the positive review summary
        d. Uses the negative reviews to generate the negative review summary
        e. Extracts product features from reviews and classifies them as positive or negative
        f. Generates a summarized product review
        g. Generates Search Engine Optimization (SEO) content

In the same way other use case can also be experimented with by selecting “New Run” and providing use case name, business profile, business rules and input data.        
'''
st.html(set1info)
st.image("architecture.png", caption="Architecture Diagram")
st.markdown(set2info, unsafe_allow_html=True)
