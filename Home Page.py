import streamlit as st

st.set_page_config(
    page_title = 'Homepage'
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



multi = '''**Problem Statement: Limitations of Non-Autonomous Agentic Framework**
Agentic frameworks have greatly complemented Generative AI by enhancing its creativity, problem-solving capabilities, adaptability and efficiency. However, a non-autonomous agentic framework lacks the ability to make independent decisions or take actions without explicit instructions. Following are some of the major limitations.
**Lack of Flexibility:** As they follow pre-defined rules and instructions, hence they work within a boundary and lack the ability to initiate actions on their own.
**Dependency on External Input:** Performance of these agents is affected if the user’s input is unclear, incorrect, incomplete or delayed.
**Limited Creativity:** These agents might struggle with tasks that require creative problem-solving as they operate within the constraints of pre-defined rules and instructions.
**Limited Personalization:** Without autonomy, these agents may struggle to provide highly personalized experiences. They can't actively gather and utilize user-specific information to tailor their responses.
**Limited Generalization:** These agents may struggle to generalize knowledge from one domain to another, as they lack the autonomy to explore and learn from new environments.
Several multi-agent frameworks are available in the industry like CrewAI, LangGraph and IBM’s BeeAI. These frameworks provide different architectures for implementing multi-agent based applications, but none of them provides a fully autonomous agents.
**Solution: Autonomous Multi-Agent Framework powered by IBM Watsonx**
This integrated framework leverages IBM watsonx's Generative AI capabilities to enable dynamic creation of autonomous worker agents based on business profiles and business rules. In our approach, a sophisticated multi-agent framework, emerges as a singular, autonomous Master Agent that orchestrates the creation of worker agents and tasks at runtime to achieve the business objectives. Our approach uses CrewAI framework as the base to implement our framework. The same approach can be extended to other frameworks.

'''
st.markdown(multi)

st.divider()
st.image("archdiag.jpg", caption="Architecture Diagram")
