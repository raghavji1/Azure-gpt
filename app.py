import os
import uuid
from datetime import datetime
from dotenv import load_dotenv, dotenv_values
from flask import Flask, request, jsonify
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI
from azure.cosmos import CosmosClient
from backend import get_embeddings_vector

# Load environment variables from .env file
if os.path.exists(".env"):
    load_dotenv(override=True)
    config = dotenv_values(".env")

# Initialize Flask app
app = Flask(__name__)

# Get environment variables related to Azure Search and OpenAI
azure_search_service_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
azure_search_service_admin_key = os.getenv("AZURE_SEARCH_SERVICE_ADMIN_KEY")
search_index_name = os.getenv("SEARCH_INDEX_NAME")
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_chat_completions_deployment_name = os.getenv("AZURE_OPENAI_CHAT_COMPLETIONS_DEPLOYMENT_NAME")

# Get environment variables related to Cosmos DB
cosmos_db_uri = os.getenv("COSMOS_DB_URI")
cosmos_db_key = os.getenv("COSMOS_DB_PRIMARY_KEY")
database_name = os.getenv("COSMOS_DB_DATABASE_ID")
container_name = os.getenv("COSMOS_DB_CONTAINER_ID")

# Initialize the Azure Search client
credential = AzureKeyCredential(azure_search_service_admin_key)
search_client = SearchClient(
    endpoint=azure_search_service_endpoint,
    index_name=search_index_name,
    credential=credential
)

# Initialize Cosmos DB client
cosmos_client = CosmosClient(cosmos_db_uri, credential=cosmos_db_key)
database = cosmos_client.get_database_client(database_name)
container = database.get_container_client(container_name)

# Function to retrieve conversation history from Cosmos DB
def get_conversation_history(user_id):
    query = "SELECT * FROM c WHERE c.user_id=@user_id ORDER BY c.timestamp DESC"
    parameters = [{"name": "@user_id", "value": user_id}]
    items = list(container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True))
    return items

# Function to format memory from conversation history
def format_memory(conversation_history):
    return "\n".join(
        [f"User: {item['user_message']}\nBot: {item['bot_response']}" for item in conversation_history[-5:]]
    )

# Function to interact with the LLM and provide answers using memory
def get_llm_response(user_input, search_content, user_id):
    # Retrieve and format conversation history as memory
    conversation_history = get_conversation_history(user_id)
    memory = format_memory(conversation_history)

    # Prepare messages for the LLM
    messages = [
        {"role": "system", "content": "You are a fantastic mechanic. You will handle all the queries about bike repairing."},
    ]
    
    if memory:
        messages.append({"role": "user", "content": memory})

    messages.append({"role": "user", "content": f"{user_input}\n\nSearch results:\n{search_content}"})

    # Interact with the LLM
    openai_client = AzureOpenAI(
        azure_endpoint=azure_openai_endpoint,
        api_key=azure_openai_api_key,
        api_version="2024-06-01"
    )
    
    response = openai_client.chat.completions.create(
        model=azure_openai_chat_completions_deployment_name,
        messages=messages
    )

    return response.choices[0].message.content

# Function to save conversation history in Cosmos DB
def save_conversation(user_id, user_message, bot_response):
    item = {
        'user_id': user_id,
        'id': str(uuid.uuid4()),  # Unique ID for the message
        'user_message': user_message,
        'bot_response': bot_response,
        'timestamp': datetime.utcnow().isoformat()  # Store timestamp
    }
    container.upsert_item(item)

def search_with_vector(query):
    embedding = get_embeddings_vector(query)  # Assuming this function is defined elsewhere
    vector_query = VectorizedQuery(vector=embedding, k_nearest_neighbors=3, fields="vector")

    results = search_client.search(
        search_text=None,
        vector_queries=[vector_query],
        select=["page_number", "page_content"]
    )

    search_results = []
    for idx, result in enumerate(results, start=1):
        page_number = result.get('page_number')
        page_content = result.get('page_content')
        search_results.append({
            "doc_ref": f"[doc{idx}]",
            "page_number": page_number,
            "page_content": page_content,
            "score": result.get('@search.score'),
            # Correctly format image path without leading '/'
            "image_path": f"output_images/{str(page_number).lower()}.jpg"
        })

    return search_results

# Function to format the search content for the LLM input
def format_search_content(search_results):
    return "\n".join(
        [f"{res['doc_ref']} page {res['page_number']}: {res['page_content']}" for res in search_results]
    )

@app.route("/", methods=["GET"])
def welcome():
    return jsonify(message="Hello, welcome to the API :)")

# Endpoint to interact with the LLM
@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    user_id = data.get('user_id')
    user_input = data.get('question')

    # Perform vector search based on the user input
    search_results = search_with_vector(user_input)

    # Format the search content to be used as context for the LLM
    formatted_content = format_search_content(search_results)

    # Get the response from the LLM
    response = get_llm_response(user_input, formatted_content, user_id)

    # Save conversation history in Cosmos DB
    save_conversation(user_id, user_input, response)

    # Send the response back to the UI with text and image paths
    return jsonify({
        'response': response,
        'images': [res['image_path'] for res in search_results]  # Return the image paths
    })

# Endpoint to retrieve conversation history
@app.route('/history/<user_id>', methods=['GET'])
def history(user_id):
    conversation_history = get_conversation_history(user_id)
    return jsonify(conversation_history)

if __name__ == "__main__":
    app.run(debug=True)