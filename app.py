from fastapi import FastAPI, Request, Form, Response, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import logging
import json
import os
from dotenv import load_dotenv
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Load environment variables
load_dotenv()

# Initialize Ollama
logger.info("Initializing Ollama LLM")
llm = OllamaLLM(
    model="cniongolo/biomistral",
    temperature=0.7,
    max_tokens=500,
)
logger.info("Ollama LLM Initialized.")

# Initialize embeddings
logger.info("Loading embeddings model")
embeddings = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

# Initialize ChromaDB
persist_directory = "chroma_db"
logger.info(f"Initializing ChromaDB with persist directory: {persist_directory}")
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
collection = db.get()
doc_count = len(collection['ids'])
logger.info(f"ChromaDB initialized. Number of documents: {doc_count}")
if doc_count == 0:
    logger.warning("ChromaDB collection is empty. Make sure to run ingest.py first.")

# Set up prompt template
prompt_template = """You are a medical assistant. Please answer the following question:

Context: {context}
Question: {question}

Provide a detailed and comprehensive answer. Explain thoroughly and give examples if applicable.
Your response should be at least 150 words long, unless the question can be fully answered in fewer words.

Answer:"""

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

# Set up retriever
retriever = db.as_retriever(search_kwargs={"k": 1})

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def generate_response(query, context):
    logger.info("Starting response generation")
    start_time = time.time()
    
    try:
        response = llm.invoke(prompt.format(context=context, question=query))
        end_time = time.time()
        logger.info(f"Response generated in {end_time - start_time:.2f} seconds")
        logger.info(f"Raw model output: {response}")
        return response
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        return None

@app.post("/get_response")
async def get_response(query: str = Form(...)):
    try:
        logger.info(f"Searching for documents related to query: {query}")
        docs = retriever.get_relevant_documents(query)
        if not docs:
            logger.warning("No relevant documents found")
            context = "No relevant documents found."
            source_document = "N/A"
            doc = "N/A"
        else:
            context = docs[0].page_content
            source_document = context
            doc = docs[0].metadata.get('source', 'N/A')
        
        logger.info("Generating response using the language model")
        
        with ThreadPoolExecutor() as executor:
            future = executor.submit(generate_response, query, context)
            try:
                response = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: future.result(timeout=240)  # 240 seconds timeout
                )
            except TimeoutError:
                return JSONResponse(
                    content={"error": "Response generation timed out after 240 seconds"},
                    status_code=504
                )

        if response is None:
            return JSONResponse(
                content={"error": "Failed to generate response"},
                status_code=500
            )

        # Extract the answer part from the response
        answer_parts = response.split("Answer:")
        if len(answer_parts) > 1:
            answer = answer_parts[-1].strip()
        else:
            answer = response.strip()

        logger.info(f"Processed answer: {answer}")

        response_data = jsonable_encoder(json.dumps({
            "answer": answer, 
            "source_document": source_document, 
            "doc": doc
        }))
        return Response(response_data)

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return JSONResponse(
            content={"error": f"An error occurred: {str(e)}"},
            status_code=500
        )

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting the server")
    uvicorn.run(app, host="127.0.0.1", port=8000)
