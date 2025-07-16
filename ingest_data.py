
import os
import json
from typing import List, Dict, Any
from langchain_core.documents import Document
from assistant.infrastructure.qdrant.service import vectorstore

def read_all_json_files(folder_path: str) -> List[Dict[str, Any]]:
    """
    Reads all JSON files in the specified folder and returns a list of dictionaries
    containing the filename and its parsed JSON content.

    Parameters:
        folder_path (str): Path to the folder containing JSON files.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries with keys 'filename' and 'data'.
    """
    result = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            full_path = os.path.join(folder_path, filename)
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                result.append(data)
            except Exception as e:
                print(f"Failed to read {filename}: {e}")

    return result

knowledge_base = read_all_json_files("knowledge_base")

processed_docs = []

for data_collection in knowledge_base:
    for doc in data_collection:
        metadata = doc['category']
        metadata = metadata.replace(" ", "_").replace("&", 'and')
        data = doc['doc']
        processed_docs.append(Document(page_content=data,
                                    metadata={"source": metadata}))

vector_store = vectorstore()

# Add documents
vector_store.add_documents(processed_docs)