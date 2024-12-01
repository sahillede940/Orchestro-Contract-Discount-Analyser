from pathlib import Path
from utils.lama_parse import extract_text_from_file
from langchain_pinecone import PineconeVectorStore
from utils.get_embedding_model import get_embedding_model
from utils.constants import PDF_INDEX_NAME


async def process_and_store_pdf_with_langchain(
    filename: Path,
    namespace: str
):

    # Extract text from PDF
    docs_to_store = await extract_text_from_file(filename)

    # Store embeddings in Pinecone
    embedding = get_embedding_model()
    vector_store = PineconeVectorStore(embedding=embedding)
    vector_store.from_documents(namespace=namespace, embedding=embedding,
                                documents=docs_to_store, index_name=PDF_INDEX_NAME)

    return {
        "file_id": filename,
        "pages_embedded": len(docs_to_store),
        "namespace": namespace,
    }
