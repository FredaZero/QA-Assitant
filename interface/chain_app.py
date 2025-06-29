import chainlit as cl
from chainlit import run_sync
from chainlit import make_async
import logging
import os
import numpy as np
from getpass import getpass
import pandas as pd
from collections import Counter
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.auto import partition
from unstructured.embed.openai import OpenAIEmbeddingConfig, OpenAIEmbeddingEncoder
from unstructured.embed.openai import OpenAIEmbeddingConfig, OpenAIEmbeddingEncoder
import fitz
from langchain_openai import OpenAIEmbeddings
import kdbai_client as kdbai
from langchain_community.vectorstores import KDBAI
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import openai
from openai import OpenAI
import nltk
nltk.download('punkt')
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if "OPENAI_API_KEY" in os.environ:
    KDBAI_API_KEY = os.environ["OPENAI_API_KEY"]
else: 
    OPENAI_API_KEY = getpass("OPENAI API KEY: ")
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# Check if KDBAI_ENDPOINT is in the environment variables
if "KDBAI_ENDPOINT" in os.environ:
    KDBAI_ENDPOINT = os.environ["KDBAI_ENDPOINT"]
else:
    # Prompt the user to enter the API key
    KDBAI_ENDPOINT = input("KDB.AI ENDPOINT: ")
    # Save the API key as an environment variable for the current session
    os.environ["KDBAI_ENDPOINT"] = KDBAI_ENDPOINT
# Check if KDBAI_API_KEY is in the environment variables

KDBAI_API_KEY = (os.environ["KDBAI_API_KEY"]
                 if "KDBAI_API_KEY" in os.environ
                 else input("KDB.AI API KEY: "))
#connect to KDB.AI
session = kdbai.Session(endpoint=KDBAI_ENDPOINT, api_key=KDBAI_API_KEY)
table = session.table("Contextualized_Table")
# use KDBAI as vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vecdb_kdbai_contextualized = KDBAI(table, embeddings)
# Define a Question/Answer LangChain chain

qabot_contextualized = RetrievalQA.from_chain_type(
    chain_type="stuff",
    llm=ChatOpenAI(model="gpt-4o"),
    retriever=vecdb_kdbai_contextualized.as_retriever(search_kwargs=dict(k=5)),
    return_source_documents=True,
)
async def QA_function(query):
    results = qabot_contextualized.invoke(dict(query=query))["result"]
    
    return results

cl.config.LANGUAGE = "en"

@cl.on_chat_start
async def start():
    
    await cl.Message(content="Welcome! Here you can ask any questions about Mata's financial reports from 2023 to July 2024").send()

@cl.on_message
async def main(message: cl.Message):
    

    result = run_sync(QA_function(message.content))
    await cl.Message(content=f"{result}").send()


