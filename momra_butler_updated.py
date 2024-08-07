import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import os
from openai import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain_community.utilities import SQLDatabase
import sqlite3
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from cryptography.fernet import Fernet

# Streamlit application layout
st.set_page_config(page_title="Momrah Butler", page_icon=":robot_face:", layout="wide")
st.title("🤖 Momrah Butler")

# Load and display the data
st.markdown("## Data Overview")
#data_df = pd.read_csv('input.csv')
#st.dataframe(data_df.head(), use_container_width=True)

from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///mydatabase_incidentshotspotscis.db")
print(db.dialect)
print(db.get_usable_table_names())
db.run("SELECT count(*) FROM aioutbound_incidents")


# Decryption function for API key
secret = 'FXsa85E_-C92Yy5MTKOvZq2habZOZ_Rg4Q825VDPUrU='
masked_key = 'gAAAAABmaXGkYa10tSB5eYllz78iOxwnTIwtK5yyP6fE2l0cLC9b75DiEKd7VXoKCKcJaNVaWoXQiyIG7XLPNZItdzkG9MAyU-xI_fcpaWX6vAoetceL3OCFVseNJUcbXJNOy79zy603nB5x30BI9N0Cn1rp0V3ZNw=='

def demask_api_key(encrypted_key, key):
    fernet = Fernet(key)
    decrypted_key = fernet.decrypt(encrypted_key).decode()
    return decrypted_key

demasked_key = demask_api_key(masked_key, secret)
os.environ["OPENAI_API_KEY"] = demasked_key

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Chain 1: Table Category Prompt
table_category_prompt = """Return any one of the below table categories that are relevant to the user question. \
The table categories are:

Hotspots
Incidents
"""

category_chain = create_extraction_chain_pydantic(TableCustom, llm, system_message=table_category_prompt)

# Chain 2: Get Tables based on Categories
def get_tables(categories: List[Table]) -> List[str]:
    #tables = []
    for category in categories:
        if category.name == "Hotspots":
            return  "ai_hotspot_metrics"
                
        elif category.name == "Incidents":
            return "ai_outbound"
        
        elif category.name=="CIS_WorkFlow":
            return "pc_momra_cis_workflow"
        
    return tables
	
def process_table_categories(prompt_response):
    print("prompt response---")
    print(prompt_response)
    categories = category_chain.invoke({"input": prompt_response})
    print("categories")
    print(categories)
    return get_tables(categories)
	
# Chain 3: Execute Query and Answer Prompt
execute_query = QuerySQLDataBaseTool(db=db)

# Define the combined prompt template for writing queries
write_query_prompt_template = PromptTemplate.from_template(
    """Given the following user question and relevant SQL tables, generate an SQL query.

Question: {question}
Tables: {tables}
SQL Query: """
)

write_query = create_sql_query_chain(llm, db)

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question. Give the response only in English.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

answer_chain = answer_prompt | llm | StrOutputParser()


def query_chain(question: str) -> str:
    # Get table names
    tables = process_table_categories(question)
    print("tables--")
    print(tables)
    
    # Generate SQL query
    query_response = write_query.invoke({"question": question, "tables": tables})
    
    # Extract the SQL query from the response
    print(query_response)
    #sql_query = query_response['output']
    
    # Execute SQL query
    result = execute_query.invoke({"query": query_response})

    print(result)
    
    # Format the answer
    answer_response = answer_chain.invoke({"question": question, "query": query_response, "result": result})
    #return answer_response['result']
    return answer_response




client = OpenAI(api_key=demasked_key)

def translate_text(text):
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Translate the following text to English: {text}",
            }
        ],
        model="gpt-3.5-turbo"
    )
    translation = response.choices[0].message.content.strip()
    return translation

def translate_text_arabic(text):
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Translate the following text to Arabic: {text}",
            }
        ],
        model="gpt-3.5-turbo"
    )
    translation = response.choices[0].message.content.strip()
    return translation

# Initialize session state for rounds, user inputs, and responses
if 'round' not in st.session_state:
    st.session_state.round = 0
    st.session_state.inputs = []
    st.session_state.responses = []

# Function to process the input and get responses
def process_input(text):
    eng_prompt = translate_text(text)
    #response = chain.invoke({"question": eng_prompt})
	response = query_chain(eng_prompt)
    arabic_response = translate_text_arabic(response)
    return response, arabic_response

# Function to create a new input field and process previous input
def handle_submit():
    # Get the input from the last round's text field
    last_input = st.session_state[f"text_input_{st.session_state.round}"]

    if last_input:
        st.session_state.inputs.append(last_input)
        st.session_state.responses.append(None)  # Placeholder for response
        st.session_state.round += 1
        st.experimental_rerun()

# Create text input field for the current round
text_input = st.text_input(
    f"Enter your Query:", key=f"text_input_{st.session_state.round}", placeholder="Type your query here..."
)

# Submit button to process the current input and create a new input field
st.markdown("### Actions")
if st.button("Submit"):
    handle_submit()

# Process the inputs and get responses for each round
for i in range(len(st.session_state.inputs)):
    if st.session_state.responses[i] is None:
        response, arabic_response = process_input(st.session_state.inputs[i])
        st.session_state.responses[i] = (response, arabic_response)

# Display the results of each round with enhanced styling
st.markdown("## Results")
for i in range(len(st.session_state.inputs)):
    st.markdown(f"### Query {i + 1}")
    st.markdown(f"**Input:** {st.session_state.inputs[i]}")
    if st.session_state.responses[i]:
        with st.expander("View Response"):
            st.markdown(f"**Response:**")
            st.write(st.session_state.responses[i][0])
            st.markdown(f"**Arabic Response:**")
            st.write(st.session_state.responses[i][1])
    st.markdown("---")

# Limiting to 10 rounds
if st.session_state.round >= 10:
    st.warning("Reached maximum number of rounds (10).")

# Styling enhancements
st.markdown(
    """
    <style>
    .stTextInput>div>div {
        background-color: #f9f9f9;
        border: 1px solid #ddd;
        border-radius: 5px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stExpander>summary {
        font-weight: bold;
        background-color: #f1f1f1;
        padding: 5px 10px;
        border: 1px solid #ddd;
        border-radius: 5px;
    }
    .stExpander>div>div {
        padding: 10px;
        background-color: #fff;
        border: 1px solid #ddd;
        border-top: none;
        border-radius: 0 0 5px 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
