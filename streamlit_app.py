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



"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:.
If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""

# OpenAI API key
open_ai_api = "sk-proj-N7m3Hr6mhKZe8BzdhMQST3BlbkFJiw5H8yFX2bmP3WnhStzK"
os.environ["OPENAI_API_KEY"] = open_ai_api

# Streamlit application layout
st.title("Momrah Butler")

os.environ["OPENAI_API_KEY"] = open_ai_api
db = SQLDatabase.from_uri("sqlite:///your_database.db")
#print(db.dialect)
#print(db.get_usable_table_names())
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

client = OpenAI(api_key=open_ai_api)

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


#num_points = st.slider("Number of points in alliswell123", 1, 10000, 1100)
#num_turns = st.slider("Number of turns in spiral", 1, 300, 31)

#indices = np.linspace(0, 1, num_points)
#theta = 2 * np.pi * num_turns * indices
#radius = indices

#x = radius * np.cos(theta)
#y = radius * np.sin(theta)

#df = pd.DataFrame({
#    "x": x,
#    "y": y,
#    "idx": indices,
#    "rand": np.random.randn(num_points),
#})

#st.altair_chart(alt.Chart(df, height=700, width=700)
#    .mark_point(filled=True)
#    .encode(
#        x=alt.X("x", axis=None),
#        y=alt.Y("y", axis=None),
#        color=alt.Color("idx", legend=None, scale=alt.Scale()),
#        size=alt.Size("rand", legend=None, scale=alt.Scale(range=[1, 150])),
 #   ))
