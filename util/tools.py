import math
from datetime import datetime, timezone
from langchain_community.agent_toolkits.openapi.toolkit import RequestsToolkit
from langchain_community.utilities.requests import TextRequestsWrapper
from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document

embeddings = OllamaEmbeddings(model="gemma3:4b")

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.

    Args:
        expression: A mathematical expression to evaluate.
                    Supports +, -, *, /, **, sqrt(), abs(), etc.
    """
    # Safe math evaluation with limited builtins
    allowed_names = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sqrt": math.sqrt,
        "pow": pow,
        "pi": math.pi,
        "e": math.e,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"

@tool
def search_documents(query: str) -> str:
    """Sök efter information i den interna dokumentsamlingen (RAG). 
    Använd detta för specifika frågor om företagsdata eller sparade dokument.
    """
    try:
        # 1. Simulate a collection of documents (In a real app, these come from PDFs/Files)
        raw_documents = [
            "Företagets policy för hemarbete tillåter 2 dagar per vecka.",
            "Projekt 'Alpha' har deadline den 15:e maj 2026.",
            "Kontaktpersonen för IT-support är Erik Andersson.",
            "Frukost serveras i matsalen klockan 08:00 varje morgon."
        ]
        
        # 2. Create the Vector Store (This usually happens once and is saved)
        text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        docs = [Document(page_content=t) for t in raw_documents]
        
        vectorstore = FAISS.from_documents(docs, embeddings)
        
        # 3. Perform the search
        results = vectorstore.similarity_search(query, k=2)
        
        context = "\n".join([res.page_content for res in results])
        return f"Hittade följande information:\n{context}"
    except Exception as e:
        return f"RAG-sökning misslyckades: {e}"
@tool
def get_current_time() -> str:
    """Get the current date and time in UTC."""
    now = datetime.now(timezone.utc)
    return f"Current UTC time: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}"

def get_web_search_tool(): 
    toolkit = RequestsToolkit(
        requests_wrapper=TextRequestsWrapper(headers={}),
        allow_dangerous_requests=True,
    )
    return toolkit.get_tools()

@tool
def read_local_file(file_path: str) -> str:
    """Använd detta verktyg för att läsa innehållet i en textfil (.txt, .md, .py) på datorn.
    Du måste ange den exakta sökvägen till filen.
    """
    try:
        # We use 'with' to ensure the file is closed properly
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            return f"Innehåll i {file_path}:\n\n{content}"
    except FileNotFoundError:
        return f"Fel: Filen hittades inte på sökvägen: {file_path}"
    except Exception as e:
        return f"Ett fel uppstod vid läsning av filen: {str(e)}"
    

@tool
def scrape_website(url: str) -> str:
    """Reads a webpage and returns only the visible text, removing all code/scripts."""
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 1. REMOVE ALL JAVASCRIPT AND CSS
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()
            
        # 2. GET TEXT FROM SPECIFIC TAGS (Like paragraphs and headings)
        # This ignores the metadata that confused your agent
        text_blocks = soup.find_all(['h1', 'h2', 'h3', 'p'])
        clean_text = "\n".join([t.get_text().strip() for t in text_blocks])
        
        # 3. LIMIT LENGTH (To keep it concise)
        return clean_text[:2000] if clean_text else "No readable text found."
    except Exception as e:
        return f"Error: {e}"