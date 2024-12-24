import os

from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# from translate import Translator
import groq

# Load environment variables for API keys
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Initialize the Flask application
app = Flask(__name__)
    
# # Initialize translators for each language
# hausa_translator = Translator(to_lang="ha")
# igbo_translator = Translator(to_lang="ig")
# yoruba_translator = Translator(to_lang="yo")
# swahili_translator = Translator(to_lang="sw")

# # Route to handle speech input and convert it to text
# @app.route('/speech_query', methods=['POST'])
# def speech_query():
#     try:
#         # Capture user input via speech
#         user_input = speech_to_text()
#         if not user_input:
#             return jsonify({'error': 'Speech not recognized'}), 400

#         # Process the user input (replace this with your chatbot logic)
#         response = f"You said: {user_input}"

#         # Convert the chatbot response into speech
#         text_to_speech(response)

#         return jsonify({'response': response})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500
    



# ---------- Combined Routes ---------- #
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/legal')
def legal():
    return render_template('legal.html')

@app.route('/civic')
def civic():
    return render_template('civic.html')

@app.route('/hausa')
def hausa():
    return render_template('hausa.html')

@app.route('/yoruba')
def yoruba():
    return render_template('yoruba.html')

@app.route('/igbo')
def igbo():
    return render_template('igbo.html')


@app.route('/civic_hausa')
def hausa_result():
    return render_template('civic_hausa.html')

@app.route('/civic_yoruba')
def yoruba_result():
    return render_template('civic_yoruba.html')

@app.route('/civic_igbo')
def igbo_result():
    return render_template('civic_igbo.html')


# Function to translate content to Hausa using Groq
GROQ_API_KEY = "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"
client = groq.Groq(api_key=GROQ_API_KEY)

def translate_to_hausa(text, model="llama-3.1-70b-versatile", temperature=0.7, max_tokens=2000):
    # Define the prompt for translation
    prompt = f"Translate the following text to Hausa: \"{text}\"."

    # Call the Groq API for translation
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that translates English to Hausa."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )

    # Extract and return the translated text
    translated_text = response.choices[0].message.content.strip()
    return translated_text

# Yoruba Translation Function
def translate_to_yoruba(text, model="llama-3.1-70b-versatile", temperature=0.7, max_tokens=2000):
    # Define the prompt for translation
    prompt = f"Translate the following text to Yoruba: \"{text}\"."

    # Call the Groq API for translation
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that translates English to Yoruba."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )

    # Extract and return the translated text
    translated_text = response.choices[0].message.content.strip()
    return translated_text


# Igbo Translation Function
def translate_to_igbo(text, model="llama-3.1-70b-versatile", temperature=0.7, max_tokens=2000):
    # Define the prompt for translation
    prompt = f"Translate the following text to Igbo: \"{text}\"."

    # Call the Groq API for translation
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that translates English to Igbo."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )

    # Extract and return the translated text
    translated_text = response.choices[0].message.content.strip()
    return translated_text



# ---------- Part 3: Groq Integration for Civic and Legal Queries ---------- #


# Function to provide civic information in Nigeria using Groq
def civic_information_assistant(text, model="llama-3.1-70b-versatile", temperature=0.7, max_tokens=2000):
    prompt = f"Provide civic information for the following query: \"{text}\" related to Nigerian law, rights, governance, or policies."
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert civic information assistant focused solely on providing accurate and concise information about Nigerian laws, civic rights, governance, and public policies. If a user's query is unrelated to civic topics, inform them politely and suggest they rephrase their question to focus on civic matters."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    civic_info = response.choices[0].message.content.strip()
    return civic_info


@app.route('/civic_hausa', methods=['POST'])
def civic_hausa():
    data = request.get_json()
    user_query = data.get('query')

    # Get civic information in English first
    civic_info = civic_information_assistant(user_query)

    # Translate the civic information to Hausa using OpenAI
    
    # print("this is civic info",civic_info)
    response = translate_to_hausa(civic_info)
    # response=hausa_translator.translate(civic_info)
    # print("this is response",response)
    

    return jsonify({'response': response})

@app.route('/civic_yoruba', methods=['POST'])
def civic_yoruba():
    data = request.get_json()
    user_query = data.get('query')

    # Get civic information in English first
    civic_info = civic_information_assistant(user_query)

    # Translate the civic information to Yoruba using OpenAI
    response = translate_to_yoruba(civic_info)

    return jsonify({'response': response})


@app.route('/civic_igbo', methods=['POST'])
def civic_igbo():
    data = request.get_json()
    user_query = data.get('query')

    # Get civic information in English first
    civic_info = civic_information_assistant(user_query)

    # Translate the civic information to Igbo using OpenAI
    response = translate_to_igbo(civic_info)

    return jsonify({'response': response})



# Function to provide legal advice using Groq
def legal_information_assistant(text, model="llama-3.1-70b-versatile", temperature=0.7, max_tokens=1000):
    prompt = f"""
    You are an AI lawyer specializing in Nigerian law, including the Nigerian Constitution and all relevant statutes.
    Your task is to provide accurate and thorough legal advice, opinions, and potential strategies to address the user's legal challenges.
    Always refer to the relevant sections of the law and provide a clear explanation of the legal context.
    If applicable, suggest the best alternative approaches and ways to potentially win the case.

    Question: \"{text}\"
    Based on the Nigerian Constitution and laws, here is your detailed answer:
    """
    
    # Call the Groq API to generate a response
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert AI legal assistant providing advice based on Nigerian law, "
                    "including the Constitution and statutes. "
                    "You must never mention or imply any knowledge cutoff date. "
                    "If the user's question is unrelated to Nigerian law or legal matters apart from greetings, politely inform them "
                    "that you only respond to questions related to Nigerian legal matters, the Constitution, and statutes. "
                    "Suggest that they rephrase their question to focus on legal issues if necessary."
                    "If the user's input is a greeting, respond with a greeting and ask how you can assist them with legal or constitutional matters."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    # Get the legal advice from the response
    legal_info = response.choices[0].message.content.strip()
    return legal_info

# Define the legal query endpoint to handle legal-related user queries
@app.route('/legal_query', methods=['POST'])
def legal_query():
    try:
        data = request.get_json()
        user_query = data.get('query')  # Extract user query from the request
        
        if not user_query:
            return jsonify({'error': 'No query provided'}), 400
        
        # Call the legal information assistant function with the user's query
        response = legal_information_assistant(user_query)
        
        return jsonify({'result': response})  # Return the response to the frontend
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route for processing civic queries
@app.route('/civic_query', methods=['POST'])
def civic_query():
    # Get JSON data from the request
    data = request.get_json()
    user_query = data.get('query')  # Extract the 'query' field from the JSON
    
    # Process the query using the civic_chain function
    response = civic_information_assistant(user_query)
    
    # Return the result as a JSON response
    return jsonify({"response": response})

# Start the Flask application
if __name__ == '__main__':
    app.run(debug=True)
