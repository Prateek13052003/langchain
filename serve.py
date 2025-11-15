from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langserve import add_routes

# Load environment variables from .env file
load_dotenv()

# Read API key from .env file
groq_api_key = os.getenv("GROQ_API_KEY")

# Use a valid Groq model (versatile models are decommissioned)
model = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=groq_api_key
)

# Prompt template
generic_template = "Translate the following language {language}:"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", generic_template),
        ("user", "{text}")
    ]
)

# Output parser
parser = StrOutputParser()

# Create the chain
chain = prompt | model | parser

# FastAPI app
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API server using langchain runnable interfaces"
)

# Add LangServe routes
add_routes(
    app,
    chain,
    path="/chain"
)

# Start server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8080)
