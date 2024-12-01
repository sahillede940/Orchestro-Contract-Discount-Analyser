from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from dotenv import load_dotenv
import os
import openai
from langchain_pinecone import PineconeVectorStore
from utils.pinecone import process_and_store_pdf_with_langchain
from utils.constants import PDF_INDEX_NAME, PDF_DIR
from utils.get_embedding_model import get_embedding_model
from utils.helper_func import convert_file_name_to_namespace
from openai import OpenAI
from pydantic import BaseModel
from typing import List
# Load environment variables
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create folder if it doesn't exist
Path(PDF_DIR).mkdir(parents=True, exist_ok=True)


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Endpoint to upload a PDF file and save it in a local folder.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400, detail="Invalid file type. Only PDFs are allowed.")

    file_path = Path(PDF_DIR) / file.filename
    # Save file to the folder
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    namespace = convert_file_name_to_namespace(file.filename)

    result = process_and_store_pdf_with_langchain(
        file_path=file_path, namespace=namespace)

    return {"message": "File uploaded successfully", "file_path": str(file_path), **result}


@app.get("/list-files/")
def list_files():
    """
    Endpoint to list all PDF files in the upload folder.
    """
    files = [f.name for f in Path(PDF_DIR).iterdir() if f.is_file()]
    return {"files": files}


class DomesticAirLevelRow(BaseModel):
    domestic_air_service_level: str
    weight_range: str
    current_ups: str

class DomesticAirLevelTable(BaseModel):
    rows: List[DomesticAirLevelRow]

@app.get("/similar-pages")
async def query_pdf(file_name: str, charges_band: str):
    """
    Query embeddings for a specific file's namespace in Pinecone.
    """
    query = """
    Use the attached contract to fill the table.
    the weekly charges band is {charges_band} DOMESTIC AIR SERVICE LEVEL
    WEIGHT RANGE CURRENT UPS
    Next Day Air Letter All
    Next Day Air Package All
    Next Day Air Saver Letter All
    Next Day Air Saver Package All
    2nd Day AM Letter All
    2nd Day AM Package All
    2nd Day Air Letter All
    2nd Day Air Package All
    3 Day Select Package All
    Next Day Air CWT All
    Next Day Air Saver CWT All
    2nd Day Air AM CWT All
    2nd Day Air CWT All
    3 Day Select CWT All
    """.format(charges_band=charges_band)

    namespace = convert_file_name_to_namespace(file_name)
    vector_store = PineconeVectorStore(
        pinecone_api_key=PINECONE_API_KEY, index_name=PDF_INDEX_NAME, embedding=get_embedding_model())
    results = vector_store.similarity_search_with_score(
        k=3, query=query, namespace=namespace)

    context = ""
    i = 0
    for result in results:  # Unpacking the tuple into result and score
        i += 1
        context += f"Page {i}: {result[0].page_content}\n\n"

    # save in text
    with open("context.txt", "w") as f:
        f.write(context)
    
    prompt = f"""
    Contract Context: 
    {context}
    
    Question: {query}
    """.format(context=context, query=query)

    try:
        messages = [
            {"role": "user", "content": prompt},
        ]
        client = OpenAI()

        completion = client.beta.chat.completions.parse(
            model="gpt-4o", messages=messages, response_format=DomesticAirLevelTable
        )
        response = completion.choices[0].message.parsed
        return response
    except Exception as e:
        return {"error": f"An error occurred while processing the request: {str(e)}"}
