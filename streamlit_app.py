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
from config import open_ai_api_key
#from dotenv import load_dotenv


"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:.
If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""

# OpenAI API key

os.environ["OPENAI_API_KEY"] = open_ai_api_key

# Streamlit application layout
st.title("Momrah Butler")
data_df=pd.read_csv('input.csv')

st.write(f"{len(data_df)}")
try:
    # Create a SQLite connection
    sqlite_conn = sqlite3.connect('your_database.db')

    # Write the DataFrame to the SQLite database
    data_df.to_sql('ai_outbound', sqlite_conn, if_exists='replace', index=False)

    # Commit and close the connection
    sqlite_conn.commit()
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    sqlite_conn.close()
secret='FXsa85E_-C92Yy5MTKOvZq2habZOZ_Rg4Q825VDPUrU='
masked_key='gAAAAABmaXGkYa10tSB5eYllz78iOxwnTIwtK5yyP6fE2l0cLC9b75DiEKd7VXoKCKcJaNVaWoXQiyIG7XLPNZItdzkG9MAyU-xI_fcpaWX6vAoetceL3OCFVseNJUcbXJNOy79zy603nB5x30BI9N0Cn1rp0V3ZNw=='
def demask_api_key(encrypted_key,key):
    #key = load_key()
    fernet = Fernet(key)
    decrypted_key = fernet.decrypt(encrypted_key).decode()
    return decrypted_key

demasked_key = demask_api_key(masked_key,secret)
print(demasked_key)
os.environ["OPENAI_API_KEY"] = demasked_key
db = SQLDatabase.from_uri("sqlite:///your_database.db")
print(db.dialect)
print(db.get_usable_table_names())
st.write(f"{db.get_usable_table_names()}")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

execute_query = QuerySQLDataBaseTool(db=db)
write_query = create_sql_query_chain(llm, db)

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

answer = answer_prompt | llm | StrOutputParser()
chain = (
    RunnablePassthrough.assign(query=write_query).assign(
        result=itemgetter("query") | execute_query
    )
    | answer
)

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

arabic_text = st.text_input("Enter some text:")
eng_prompt=translate_text(arabic_text)
response=chain.invoke({"question": eng_prompt})
arabic_response=translate_text_arabic(response)

st.write(f"{response}")
st.write("\n\n")
st.write(f"{arabic_response}")
#st.write(f"Text entered: {text_input}")
