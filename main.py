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


@app.post("/upload-contract")
async def upload_contract(file: UploadFile = File(...)):
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

    result = await process_and_store_pdf_with_langchain(
        filename=file.filename, namespace=namespace)

    return {"message": "File uploaded successfully", "file_path": str(file_path), **result}



class DomesticAirLevelRow(BaseModel):
    domestic_air_service_level: str
    weight_range: str
    current_ups: str

class DomesticAirLevelTable(BaseModel):
    rows: List[DomesticAirLevelRow]

class DomesticAirLevelRequest(BaseModel):
    filename: str
    charges_band: str

@app.post("/query-contract")
async def query_pdf(request: DomesticAirLevelRequest):
    """
    Query embeddings for a specific file's namespace in Pinecone.
    """
    
    filename = request.filename
    charges_band = request.charges_band
    
    query = """
Use the attached contract to fill the table.
The weekly charges band is {charges_band} DOMESTIC AIR SERVICE LEVEL
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

    
    query += "\n If you did not find the some information in the contract, please strictly use null to fill the table."

    namespace = convert_file_name_to_namespace(filename)
    vector_store = PineconeVectorStore(
        pinecone_api_key=PINECONE_API_KEY, index_name=PDF_INDEX_NAME, embedding=get_embedding_model())
    results = vector_store.similarity_search_with_score(
        k=3, query=query, namespace=namespace)

    context = ""
    i = 0
    for result in results:
        i += 1
        context += f"Page {i}: {result[0].page_content}\n\n"
    
    prompt = """
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
