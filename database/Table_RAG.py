import os
import numpy as np
from getpass import getpass
import pandas as pd
from collections import Counter
import json
import csv
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.auto import partition
from unstructured.staging.base import elements_to_json, elements_from_json
from unstructured.staging.base import convert_to_dict
from unstructured.documents.elements import Title, NarrativeText, Table
from unstructured.staging.base import convert_to_csv
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
# nltk.download('punkt')
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if "OPENAI_API_KEY" in os.environ:
    KDBAI_API_KEY = os.environ["OPENAI_API_KEY"]
else: 
    OPENAI_API_KEY = getpass("OPENAI API KEY: ")
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

docs = partition_pdf(filename='./cpi_09162009.pdf',
                     strategy='ocr_only',
                     chunking_strategy="by_title",
                     max_characters=2800,
                     new_after_n_chars=2500,
                     
                     ) 
# infer_table_structure=True,
#                      strategy="hi_res",
#                      model_name = "yolox"
# print(Counter(type(doc) for doc in docs))
output = 'outputs.json'
elements_to_json(docs, filename=output)
elements = elements_from_json(filename=output)
count = 1
for doc in docs:
    if doc.to_dict()['type'] == 'Table':
        
        
      
       
        break
        print(doc.text)

# embed extracted elements with OpenAI embedding model
embedding_encoder = OpenAIEmbeddingEncoder(
    config=OpenAIEmbeddingConfig(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )
)
elements = embedding_encoder.embed_documents(elements=docs)

# store original elements in a dataframe
data = []
for ele in elements:
    row = {}
    row['id'] = ele.id
    row['text'] = ele.text
    row['metadata'] = ele.metadata.to_dict()
    row['embedding'] = ele.embeddings
    data.append(row)
df_non_contextualized = pd.DataFrame(data)

# create contextualized descriptions and markdown formatted tables
client = OpenAI(api_key=openai_api_key)
def get_table_description(table_content):
    prompt = f"""
    Given the following table content,
    provide the table in csv format.

    Table Content:
    {table_content}

    Please provide only the table in csv format.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that formats table in csv."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as f:
        for page in f:
            text += page.get_text()
    return text

pdf_path = './cpi_09162009.pdf'
# document_content = extract_text_from_pdf(pdf_path)

# Process each table in the directory
for doc in docs:
  if doc.to_dict()['type'] == 'Table':
    table_content = doc.to_dict()['text']

    # Get description and markdown table from GPT-4
    result = get_table_description(table_content)
    doc.text = result
    print(doc.text)
    with open('data_file1.csv', 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(doc.text)
    break

print("Processing complete.")

elements = embedding_encoder.embed_documents(
    elements=docs
)

data = []

for c in elements:
  row = {}
  row['id'] = c.id
  row['text'] = c.text
  row['metadata'] = c.metadata.to_dict()
  row['embedding'] = c.embeddings
  data.append(row)

df_contextualized = pd.DataFrame(data)

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
# create schema and KDB.AI table
schema = {'columns': [
         {'name': 'id', 'pytype': 'str'},
         {'name': 'text', 'pytype': 'str'},
         {'name': 'metadata', 'pytype': 'dict'},
         {'name': 'embedding',
             'vectorIndex': {'dims': 1536, 'type': 'flat', 'metric': 'L2'}}]}

Contextualized_KDBAI_TABLE_NAME = "Contextualized_Table"
non_Contextualized_KDBAI_TABLE_NAME = "Non_Contextualized_Table"

# First ensure the tables do not already exist
if Contextualized_KDBAI_TABLE_NAME in session.list():
    session.table(Contextualized_KDBAI_TABLE_NAME).drop()

if non_Contextualized_KDBAI_TABLE_NAME in session.list():
    session.table(non_Contextualized_KDBAI_TABLE_NAME).drop()

#Create the tables
table_contextualized = session.create_table(Contextualized_KDBAI_TABLE_NAME, schema)
table_non_contextualized = session.create_table(non_Contextualized_KDBAI_TABLE_NAME, schema)
# Insert Elements into the KDB.AI Tables
table_contextualized.insert(df_contextualized)
table_non_contextualized.insert(df_non_contextualized)

# use KDBAI as vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vecdb_kdbai_contextualized = KDBAI(table_contextualized, embeddings)
vecdb_kdbai_non_contextualized = KDBAI(table_non_contextualized, embeddings)

# Define a Question/Answer LangChain chain
qabot_contextualized = RetrievalQA.from_chain_type(
    chain_type="stuff",
    llm=ChatOpenAI(model="gpt-4o"),
    retriever=vecdb_kdbai_contextualized.as_retriever(search_kwargs=dict(k=5)),
    return_source_documents=True,
)

qabot_non_contextualized = RetrievalQA.from_chain_type(
    chain_type="stuff",
    llm=ChatOpenAI(model="gpt-4o"),
    retriever=vecdb_kdbai_non_contextualized.as_retriever(search_kwargs=dict(k=5)),
    return_source_documents=True,
)
# Helper function to perform RAG
def RAG(query):
  print(query)
  print("-----")
  print("Contextualized")
  print("-----")
  print(qabot_contextualized.invoke(dict(query=query))["result"])
  print("-----")
  print("Non Contextualized")
  print("-----")
  print(qabot_non_contextualized.invoke(dict(query=query))["result"])

if __name__ == "__main__":
   RAG("What is the research and development costs for six months ended in June 2024")