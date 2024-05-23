_import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from openai import OpenAI
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
# Title of the app
st.title("Sivaram's personal mistral with rag pipeline")
from sklearn.metrics.pairwise import cosine_similarity



with open('./config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
config['credentials'],
config['cookie']['name'],
config['cookie']['key'],
config['cookie']['expiry_days'],
config['preauthorized']
)
authenticator.login()
if st.session_state["authentication_status"]:
    authenticator.logout()
    st.write(f'Welcome *{st.session_state["name"]}*')
    client = OpenAI(api_key="jhjhjh1234", base_url="http://us01odc-sv7-1-pbm586:8085/v1")

    chroma_client = chromadb.HttpClient(host='us01odc-sv7-1-pbm586', port=8000)
    embedder = SentenceTransformer('all-mpnet-base-v2')  # Or your preferred model
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")
    collection = chroma_client.get_collection(name="Chime_data", embedding_function=sentence_transformer_ef)

    # Chroma DB - Retriever Function
    def retrieve_from_chroma(query, top_k=5):
        #collection = chroma_client.collection("Chime_data")
        search_results=collection.query(
        query_texts=[query],
        n_results=5
        )
        retrieved_documents = []
        for result in search_results["documents"][0]:
            retrieved_documents.append({
                "text": result
            })
        return retrieved_documents

    def calculate_similarity(prompt, document_text):
        prompt_embedding = embedder.encode(prompt)
        document_embedding = embedder.encode(document_text)

        # Cosine similarity
        similarity = cosine_similarity([prompt_embedding], [document_embedding])[0][0]
        return similarity


    # Check if openai_model and messages are not in session state and initialize them
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "mistral"

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.aug_messages=[]

    # Loop through existing messages and display them
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Prompt user for input
 # Prompt user for input
    prompt = st.chat_input("Pass your prompt here")


    # If user input is provided
    if prompt:
        context_string=""
        # Display user input in chat interface
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)
        similarity_threshold = 0.38
        for doc in retrieve_from_chroma(prompt):
            similarity = calculate_similarity(prompt, doc['text'])
            if similarity > similarity_threshold:
                context_string += f"{doc['text']}\n"
            # ... (Optionally add metadata) ..
        #sys_prompt="[INST] Respond with concise and informative answers. Avoid phrases like 'based on the given context'. Present yourself as a knowledgeable expert, but do not explicitly state that you have referred to external sources. "
        augmented_prompt = "\n[QUERY]  "+prompt + "\n[CONTEXT] " + context_string +"\n[INST]  Dont use and mention 'based on the given context' in your response if the context is not empty and don't mention that you got a document as an input , you are an subject matter expert"
        #augmented_prompt = prompt + "\n[CONTEXT] " + context_string
        st.session_state.aug_messages.append({"role": "user", "content": augmented_prompt})

        # Get LLM response based on user input
        response = client.chat.completions.create(
        model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.aug_messages
            ],
            stream=True,
        )
        # Display LLM response in chat interface
        with st.chat_message("assistant"):
            completed_message = ""
            message = st.empty()
            # Streaming the response out
            for chunk in response:
                # If the value is not None, append it to the completed message
                if chunk.choices[0].delta.content is not None:
                    completed_message += chunk.choices[0].delta.content
                    # Update the message in the chat interface
                    message.markdown(completed_message)

        # Add LLM response to session state messages
        st.session_state.messages.append({"role": "assistant", "content": completed_message})
        st.session_state.aug_messages.append({"role": "assistant", "content": completed_message})

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')

