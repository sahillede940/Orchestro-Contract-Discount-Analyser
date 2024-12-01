from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from langchain.schema import Document
from utils.constants import PDF_DIR

# set up parser
parser = LlamaParse(
    result_type="text"  # "markdown" and "text" are available
)

file_extractor = {".pdf": parser}


async def extract_text_from_file(file_name: str):
    """
    Extract text from a file.
    """
    file_path = f"{PDF_DIR}/{file_name}"
    documents = await SimpleDirectoryReader(input_files=[file_path], file_extractor=file_extractor).aload_data()
    docs_to_store = []
    i = 0
    for page in documents:
        i += 1
        document = Document(
            page_content=page.text,
            metadata={"file_id": page.metadata["file_name"], "page_number": i}
        )
        docs_to_store.append(document)
    
    return docs_to_store