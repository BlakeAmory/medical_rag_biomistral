import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

try:
    embeddings = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
    logger.info("Embeddings model loaded successfully")
except Exception as e:
    logger.error(f"Error loading embeddings model: {str(e)}")
    raise

def load_single_document(file_path):
    try:
        loader = UnstructuredFileLoader(file_path)
        return loader.load()
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {str(e)}")
        return []

try:
    data_dir = 'data/'
    documents = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.pdf'):
            file_path = os.path.join(data_dir, filename)
            documents.extend(load_single_document(file_path))
    
    logger.info(f"Loaded {len(documents)} documents")
    
    if not documents:
        raise ValueError("No documents were successfully loaded.")

except Exception as e:
    logger.error(f"Error loading documents: {str(e)}")
    raise

text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)
texts = text_splitter.split_documents(documents)
logger.info(f"Split documents into {len(texts)} text chunks")

# Initialize and populate ChromaDB
persist_directory = "chroma_db"

try:
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectordb.persist()
    logger.info(f"Created ChromaDB collection with {len(texts)} documents")
except Exception as e:
    logger.error(f"Error creating ChromaDB collection: {str(e)}")
    raise

# Verify the number of documents in the collection
try:
    collection = vectordb.get()
    doc_count = len(collection['ids'])
    logger.info(f"Verified {doc_count} documents in the ChromaDB collection")
except Exception as e:
    logger.error(f"Error verifying document count: {str(e)}")

print("Vector DB Successfully Created!")
