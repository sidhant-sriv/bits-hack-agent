from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_core.documents import Document
import json


def flatten_form_data(json_response):
    """
    Takes a JSON response containing form data and flattens it to a simplified structure.

    Args:
        json_response: JSON data containing form structure

    Returns:
        dict: Flattened form data structure
    """
    data = (
        json_response if isinstance(json_response, dict) else json.loads(json_response)
    )

    # Create flattened structure
    flattened_data = {
        "formId": data["data"]["formId"],
        "formTitle": data["data"]["title"],
        "formDescription": data["data"]["description"],
    }

    # Add each field with empty values
    for field in data["data"]["fields"]:
        flattened_data[field["label"]] = ""

    return flattened_data


from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_core.documents import Document
import json

# Initialize embedding model and vector store
embedding_model = CohereEmbeddings(
    model="embed-english-v3.0", client=None, async_client=None
)


def store_form_data(json_data):
    """
    Store form data in the vector database

    Args:
        json_data: JSON data containing form structure

    Returns:
        str: Status message
    """
    # Extract title and description for embedding
    title = json_data["data"]["title"]
    description = json_data["data"]["description"]

    # Create Document objects
    documents = [
        Document(
            page_content=title,
            metadata={"source": "title", "full_data": json.dumps(json_data)},
        ),
        Document(
            page_content=description,
            metadata={"source": "description", "full_data": json.dumps(json_data)},
        ),
    ]

    # Add to vector store
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        collection_name="form-data",
        persist_directory="chroma.db",
    )
    vectorstore.persist()

    return "Form data stored successfully in vector database"


def query_form_data(query_text, n_results=1):
    """
    Query the vector database for similar form data

    Args:
        query_text: The query text to search for
        n_results: Number of results to return

    Returns:
        list: Retrieved form data
    """
    # Initialize the vector store
    vectorstore = Chroma(
        embedding_function=embedding_model,
        collection_name="form-data",
        persist_directory="chroma.db",
    )

    # Search the vector store
    results = vectorstore.similarity_search(query_text, k=n_results)

    # Extract and return the JSON data
    retrieved_data = []
    for doc in results:
        try:
            # Parse the JSON from metadata
            full_data = json.loads(doc.metadata["full_data"])
            retrieved_data.append({"data": full_data})
        except Exception as e:
            retrieved_data.append({"error": str(e), "content": doc.page_content})

    return retrieved_data


def process_and_store_json_file(file_path):
    """
    Load JSON data from a file, process it, and store it in the vector database

    Args:
        file_path: Path to the JSON file

    Returns:
        str: Status message
    """
    try:
        with open(file_path, "r") as file:
            json_data = json.load(file)

        result = store_form_data(json_data)
        return result
    except Exception as e:
        return f"Error processing file: {str(e)}"
