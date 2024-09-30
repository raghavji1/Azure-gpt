import os
import re
import uuid
import fitz  # PyMuPDF
from dotenv import load_dotenv, dotenv_values
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    ComplexField,
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticSearch,
    SemanticField
)

# Load environment variables
if os.path.exists(".env"):
    load_dotenv(override=True)
    config = dotenv_values(".env")

azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_chat_completions_deployment_name = os.getenv("AZURE_OPENAI_CHAT_COMPLETIONS_DEPLOYMENT_NAME")
azure_openai_embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")
embedding_vector_dimensions = int(os.getenv("EMBEDDING_VECTOR_DIMENSIONS", 1536))

azure_search_service_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
azure_search_service_admin_key = os.getenv("AZURE_SEARCH_SERVICE_ADMIN_KEY")
search_index_name = os.getenv("SEARCH_INDEX_NAME")

# OpenAI client
openai_client = AzureOpenAI(
    azure_endpoint=azure_openai_endpoint,
    api_key=azure_openai_api_key,
    api_version="2024-06-01"
)

# Initialize Search Client
credential = AzureKeyCredential(azure_search_service_admin_key)
search_client = SearchClient(
    endpoint=azure_search_service_endpoint, 
    index_name=search_index_name, 
    credential=credential
)

# Initialize SearchIndexClient
search_index_client = SearchIndexClient(
    endpoint=azure_search_service_endpoint, 
    credential=credential
)

# Function to create the index if it doesn't exist
def create_index_if_not_exists():
    try:
        # Check if index exists
        search_index_client.get_index(search_index_name)
        print(f"Index '{search_index_name}' already exists.")
    except Exception:
        # If the index does not exist, create it
        print(f"Index '{search_index_name}' not found. Creating a new index...")

        fields = [
            SimpleField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
                sortable=True,
                filterable=True,
                facetable=True,
            ),
            SearchableField(name="page_number", type=SearchFieldDataType.String),
            SearchableField(name="page_content", type=SearchFieldDataType.String),
            SearchField(name="vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                        searchable=True,
                        vector_search_dimensions=embedding_vector_dimensions,  # Adjust based on your model
                        vector_search_profile_name="myHnswProfile"
                        )
        ]

        # Configure vector search
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="myHnsw"
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name="myHnswProfile",
                    algorithm_configuration_name="myHnsw",
                )
            ]
        )

        # Configure semantic search
        semantic_config = SemanticConfiguration(
            name="my-semantic-config",
            prioritized_fields=SemanticPrioritizedFields(
                title_field=SemanticField(field_name="page_number"),
                content_fields=[SemanticField(field_name="page_content")]
            )
        )

        # Set up the index with vector search and semantic search
        search_index = SearchIndex(
            name=search_index_name, 
            fields=fields,
            vector_search=vector_search, 
            semantic_search=SemanticSearch(configurations=[semantic_config])
        )

        # Create the index
        result = search_index_client.create_index(search_index)
        print(f"Index '{result.name}' successfully created.")

# Function to get embeddings for a text
def get_embeddings_vector(text):
    response = openai_client.embeddings.create(
        input=text,
        model=azure_openai_embedding_model
    )
    embedding = response.data[0].embedding
    return embedding

# PDF text and image extraction (page-wise)
def extract_text_and_images_from_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    content_by_page = {}

    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        blocks = page.get_text("blocks")
        page_content = ""
        
        for block in sorted(blocks, key=lambda b: (b[1], b[0])):  # Sort blocks for column-wise reading
            page_content += block[4] + "\n"

        # Store the full page content without chunking
        content_by_page[f"Page_{page_number + 1}"] = page_content

    pdf_document.close()
    return content_by_page

# Process a single PDF file (without chunking, whole page)
def process_pdf(input_path):
    # Ensure index exists before proceeding
    create_index_if_not_exists()

    # Extract text and images from the PDF
    content_by_page = extract_text_and_images_from_pdf(input_path)
    
    for page, content in content_by_page.items():
        # Create a document for each page
        document = {
            "id": str(uuid.uuid4()),
            "page_number": page,
            "page_content": content,
            "vector": get_embeddings_vector(content),  # Generate embedding for the whole page
        }

        # Upload the document to Azure Search
        result = search_client.upload_documents(documents=[document])
        print(f"Uploaded {page}: {result[0].succeeded}")

# Example usage
# input_path = 'data/client.pdf'  # Replace with the actual path to your PDF
# process_pdf(input_path)
