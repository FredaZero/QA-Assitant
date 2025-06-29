import os
import pickle  # 可以使用其他序列化方法，如joblib
from llama_index.core import SimpleDirectoryReader, ServiceContext, VectorStoreIndex, Document

from qdrant_client import QdrantClient
from llama_index.llms.ollama import Ollama
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
# from llmsherpa.readers import LayoutPDFReader
# from llama_index.readers.smart_pdf_loader import SmartPDFLoader
from llama_index.core.node_parser import TokenTextSplitter  # 导入 TokenTextSplitter
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
# bring in our LLAMA_CLOUD_API_KEY
from dotenv import load_dotenv
load_dotenv()

class ChatOllama:
    def __init__(self):
        self._qdrant_url = "http://localhost:6333"
        self._client = QdrantClient(url=self._qdrant_url, prefer_grpc=False)
        self._llm = Ollama(model='mistral:7b-instruct', temperature=0.2, request_timeout=150.0, similarity_top_k=20)

        # build HuggingFaceEmbedding model
        self._embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
      

        # transmit embed_model into Settings
        Settings.embed_model = self._embed_model
#         OpenAIEmbedding(
#     model="text-embedding-3-small", embed_batch_size=100
# )
        # Settings.llm = self._llm

        # Define the parameters for the new collection
        self._collection_name = "diet_consultant_0801_db"
       
        self._index = None
        self._db_cache_path = 'knowledgebase_cache.pkl'  # 缓存文件路径
        self._create_kb()
        self._create_chat_engine()

    def _create_chat_engine(self):
        memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
        self._chat_engine = self._index.as_chat_engine(
            chat_mode="context",
            memory=memory,
            system_prompt=self._prompt,
        )

    def _create_kb(self):
        if os.path.exists(self._db_cache_path):
            with open(self._db_cache_path, 'rb') as f:
                self._index = pickle.load(f)
            print("Knowledge loaded from cache!")
        else:
            try:
                # build ServiceContext，set Ollama instance and embed_model
                # llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
                # pdf_reader = LayoutPDFReader(llmsherpa_api_url)
                # set up parser
                # Check if the file exists
                pdf_path = r'./Diet_Programme.pdf'
                if not os.path.exists(pdf_path):
                    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

                parser = LlamaParse(
                    result_type="markdown"  # "markdown" and "text" are available
                )

                # use SimpleDirectoryReader to parse our file
                file_extractor = {".pdf": parser}
                documents = SimpleDirectoryReader(input_files=[pdf_path], file_extractor=file_extractor).load_data()

                
                # Read and chunk the PDF
                
                chunks = self._chunk_documents(documents, chunk_size=1024)  # 使用自定义的 chunk size

                self._service_context = ServiceContext.from_defaults(llm=self._llm, embed_model=self._embed_model)
                
                vector_store = QdrantVectorStore(client=self._client, collection_name=self._collection_name)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                self._index = VectorStoreIndex.from_documents(
                    chunks, service_context=self._service_context, storage_context=storage_context
                )

                # Cache the created index to a file
                with open(self._db_cache_path, 'wb') as f:
                    pickle.dump(self._index, f)
                print("Knowledge created and cached successfully!")
            except Exception as e:
                print(f"Error while creating knowledgebase: {e}")

    def _chunk_documents(self, documents, chunk_size=1024, chunk_overlap=20, separator=" "):
        """Split documents into smaller chunks using TokenTextSplitter."""
        splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=separator)
        chunks = []
        for doc in documents:
            text_chunks = splitter.split_text(doc.text)  # 修改为使用doc.text访问文本
            for i, chunk in enumerate(text_chunks):
                # 使用 Document 对象保持兼容性
                chunks.append(Document(text=chunk, metadata=doc.metadata, doc_id=f"{doc.doc_id}_{i}"))
        return chunks

    def interact_with_llm(self, query):
        AgentResponse = self._chat_engine.chat(query)
        answer = AgentResponse.response
        return answer
    
    @property
    def _prompt(self):
        return """
You are a professional AI assistant consultant working on providing guidance on diet plan and weight loss plan based on customer's body type. To know customer's body type, you need to ask questions mentioned inside square brackets from customer. DON'T ASK THESE QUESTIONS 
            MULTIPLE TIMES! Only ask these question at the beginning of the conversation.

            [which one of the following best describe your body shape: Ectomorph, Endomorph, Mesomorph. what's your fat-storing area: Stomach area, Lower-body, Everywhere.]

            You should look up customer's body type from the table below based on the answers to the questions in square brackets and remember customer's body type during the conversation.
            [type A: [Ectomorph, Stomach-area], type B: [Ectomorph, Lower-body], type C: [Ectomorph, Everywhere], type D: [Endomorph, Stomach-area],
            type E: [Endomorph, Lower-body], type F: [Endomorph, Everywhere], type G: [Mesomorph, Stomach-area], type H: [Mesomorph, Lower-body], type I: [Mesomorph, Everywhere]]

    Instruction: Use the previous chat history, or the context, to interact and help the user. If you don't know the answer, just say that you don't know, don't try to make up an answer.Provide concise and short answers"""

