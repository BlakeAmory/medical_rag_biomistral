import logging
import time
from langchain_ollama import OllamaLLM

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_llm():
    logger.info("Initializing Ollama LLM")
    llm = OllamaLLM(
        model="cniongolo/biomistral",
        temperature=0.5,
        max_tokens=4096,
    )
    logger.info("Ollama LLM Initialized.")
    return llm

def test_model():
    logger.info("Starting Ollama model test...")

    llm = setup_llm()

    # Test the model with a sample question
    test_question = "What is AIDS?"
    
    logger.info("Generating response...")
    start_time = time.time()

    try:
        response = llm.invoke(test_question)
        end_time = time.time()
        logger.info(f"Response generated in {end_time - start_time:.2f} seconds")
        
        # Print the full response
        print("\nGenerated Response:")
        print(response)
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")

    logger.info("Model test completed.")

if __name__ == "__main__":
    test_model()
