from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain_pinecone import PineconeVectorStore
from utils.get_embedding_model import get_embedding_model
from utils.constants import PDF_INDEX_NAME

def process_and_store_pdf_with_langchain(
    file_path: Path,
    namespace: str
):

    # Load the PDF into LangChain's Document format
    loader = PyPDFLoader(str(file_path))
    pages = loader.load()

    # Process each page and store embeddings
    docs_to_store = []
    for i, page in enumerate(pages):
        document = Document(
            page_content=page.page_content,
            metadata={"file_id": file_path.name, "page_number": i}
        )
        docs_to_store.append(document)

    # Store embeddings in Pinecone
    embedding = get_embedding_model()
    vector_store = PineconeVectorStore(embedding=embedding)
    vector_store.from_documents(namespace=namespace, embedding=embedding, documents=docs_to_store, index_name=PDF_INDEX_NAME)

    return {
        "file_id": file_path.name,
        "pages_embedded": len(docs_to_store),
        "namespace": namespace,
    }
