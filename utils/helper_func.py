def convert_file_name_to_namespace(file_name: str) -> str:
    return file_name.replace(".pdf", "").replace(" ", "_")