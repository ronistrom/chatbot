import streamlit as st
import pandas as pd
import os
import re
import pickle
import jwt

from dotenv import load_dotenv
from langchain import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import CubeSemanticLoader
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import ChatMessage
from pathlib import Path

from utils import (
    check_input,
    log,
    call_sql_api,
    CUBE_SQL_API_PROMPT,
    _NO_ANSWER_TEXT,
)

load_dotenv()

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

def ingest_cube_meta():
    security_context = {}
    token = jwt.encode(security_context, os.environ["CUBE_API_SECRET"], algorithm="HS256")

    loader = CubeSemanticLoader(os.environ["CUBE_API_URL"], token)
    documents = loader.load()

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save vectorstore
    with open("vectorstore.pkl", "wb") as f:
        vectorstore.save_local("faiss_store")
        
if not Path("vectorstore.pkl").exists():
    with st.spinner('Loading context from semantic layer...'):
        ingest_cube_meta();

#st.title("Generative BI demo")

#multi = '''
#You can use these sample questions to quickly test the demo:
#* What is the total value of won deals in Q4 2023?
#* What's the split of won deals in 2023 between business areas?
#* What's the split of won deals in 2023 between deal types?
#* What were the top 5 biggest won deals in 2023? Output deal name, close date, value, business area and deal type?
#'''
#st.markdown(multi)

if "messages" not in st.session_state:
    st.session_state["messages"] = [ChatMessage(role="assistant", content="How can I help you?")]

for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

if question := st.chat_input():
    st.chat_message("user").write(question)

    llm = ChatOpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"), streaming=False, temperature=0, verbose=False)
    with open("vectorstore.pkl", "rb") as f:
        vectorstore = FAISS.load_local("faiss_store", OpenAIEmbeddings())

    docs = vectorstore.similarity_search(question)
    # take the first document as the best guess
    table_name = docs[0].metadata["table_name"]

    # Columns
    columns_question = "All available columns"
    column_docs = vectorstore.similarity_search(
        columns_question, filter=dict(table_name=table_name), k=15
    )

    lines = []
    for column_doc in column_docs:
        column_title = column_doc.metadata["column_title"]
        column_name = column_doc.metadata["column_name"]
        column_data_type = column_doc.metadata["column_data_type"]
        print(column_name)
        lines.append(
            f"title: {column_title}, column name: {column_name}, datatype: {column_data_type}, member type: {column_doc.metadata['column_member_type']}"
        )
    columns = "\n\n".join(lines)

    # Construct the prompt
    prompt = CUBE_SQL_API_PROMPT.format(
        input_question=question,
        table_info=table_name,
        columns_info=columns,
        top_k=1000,
        no_answer_text=_NO_ANSWER_TEXT,
    )

    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    response = llm(st.session_state.messages)
    sql_query = response.content
    columns, rows = call_sql_api(sql_query)
    df = pd.DataFrame(rows, columns=columns)
    with st.chat_message("assistant"):
        st.table(df)