import os
import time
import logging
import ray
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_together import Together
from faiss import IndexFlatL2  # Assuming using L2 distance for simplicity

# Initialize Ray
ray.init()

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Streamlit Page Configuration
st.set_page_config(page_title="BharatLAW", layout="centered")

# Display the logo image
col1, col2, col3 = st.columns([1, 30, 1])
with col2:
    st.image("https://github.com/Nike-one/BharatLAW/blob/master/images/banner.png?raw=true", use_column_width=True)

def hide_hamburger_menu():
    st.markdown("""
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)

hide_hamburger_menu()

# Initialize session state for messages and memory
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)

@st.cache_resource
def load_embeddings():
    """Load and cache the embeddings model."""
    return HuggingFaceEmbeddings(model_name="law-ai/InLegalBERT")

# Load FAISS index and embeddings
embeddings = load_embeddings()
db = FAISS.load_local("ipc_embed_db", embeddings, allow_dangerous_deserialization=True)
db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Define the prompt template
prompt_template = """
<s>[INST]
As a legal chatbot specializing in the Indian Penal Code, you are tasked with providing highly accurate and contextually appropriate responses. Ensure your answers meet these criteria:
- Respond in a bullet-point format to clearly delineate distinct aspects of the legal query.
- Each point should accurately reflect the breadth of the legal provision in question, avoiding over-specificity unless directly relevant to the user's query.
- Clarify the general applicability of the legal rules or sections mentioned, highlighting any common misconceptions or frequently misunderstood aspects.
- Limit responses to essential information that directly addresses the user's question, providing concise yet comprehensive explanations.
- Avoid assuming specific contexts or details not provided in the query, focusing on delivering universally applicable legal interpretations unless otherwise specified.
- Conclude with a brief summary that captures the essence of the legal discussion and corrects any common misinterpretations related to the topic.

CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
- [Detail the first key aspect of the law, ensuring it reflects general application]
- [Provide a concise explanation of how the law is typically interpreted or applied]
- [Correct a common misconception or clarify a frequently misunderstood aspect]
- [Detail any exceptions to the general rule, if applicable]
- [Include any additional relevant information that directly relates to the user's query]
</s>[INST]
"""

# Set up the prompt template and model for conversation
prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question', 'chat_history'])
api_key = os.getenv('TOGETHER_API_KEY')
llm = Together(model="mistralai/Mixtral-8x22B-Instruct-v0.1", temperature=0.5, max_tokens=1024, together_api_key=api_key)

qa = ConversationalRetrievalChain.from_llm(llm=llm, memory=st.session_state.memory, retriever=db_retriever, combine_docs_chain_kwargs={'prompt': prompt})

def extract_answer(full_response):
    """Extracts the answer from the LLM's full response by removing the instructional text."""
    answer_start = full_response.find("Response:")
    if answer_start != -1:
        answer_start += len("Response:")
        answer_end = len(full_response)
        return full_response[answer_start:answer_end].strip()
    return full_response

def reset_conversation():
    st.session_state.messages = []
    st.session_state.memory.clear()

# Display the conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Input for chat
input_prompt = st.chat_input("Say something...")
if input_prompt:
    with st.chat_message("user"):
        st.markdown(f"**You:** {input_prompt}")

    st.session_state.messages.append({"role": "user", "content": input_prompt})
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking 💡..."):
            result = qa.invoke(input=input_prompt)
            message_placeholder = st.empty()
            answer = extract_answer(result["answer"])

            # Initialize the response message
            full_response = "⚠️ **_Gentle reminder: We generally ensure precise information, but do double-check._** \n\n\n"
            for chunk in answer:
                # Simulate typing by appending chunks of the response over time
                full_response += chunk
                time.sleep(0.02)  # Adjust the sleep time to control the "typing" speed
                message_placeholder.markdown(full_response + " |", unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Reset button for the chat
    if st.button('🗑️ Reset All Chat', on_click=reset_conversation):
        st.experimental_rerun()

# Footer display function
def layout(*args):
    # Custom CSS to hide the Streamlit footer and menu
    style = """
    <style>
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
     .stApp { bottom: 40px; }
    </style>
    """
    
    st.markdown(style, unsafe_allow_html=True)

    # Add your custom content to the layout
    for arg in args:
        if isinstance(arg, str):
            st.markdown(arg)
        elif isinstance(arg, HtmlElement):
            st.markdown(str(arg), unsafe_allow_html=True)

def footer():
    # Add footer content
    myargs = [
        "Made with ❤️ by Nikhil, Mihir, Nilay",
    ]
    layout(*myargs)

# Call footer function to display it
footer()

# Load and process documents
logging.info("Loading documents...")
loader = DirectoryLoader('data', glob="./*.txt")
documents = loader.load()

# Extract text and split it
logging.info("Extracting and splitting texts from documents...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
texts = []
for document in documents:
    text_content = document.get_text() if hasattr(document, 'get_text') else ""
    texts.extend(text_splitter.split_text(text_content))

# Define embedding function
def embedding_function(text):
    embeddings_model = HuggingFaceEmbeddings(model_name="law-ai/InLegalBERT")
    return embeddings_model.embed_query(text)

# Create FAISS index
index = IndexFlatL2(768)  # Dimension of embeddings, adjust as needed
docstore = {i: text for i, text in enumerate(texts)}
index_to_docstore_id = {i: i for i in range(len(texts))}
faiss_db = FAISS(embedding_function, index, docstore, index_to_docstore_id)

# Process and store embeddings in FAISS
logging.info("Storing embeddings in FAISS...")
for i, text in enumerate(texts):
    embedding = embedding_function(text)
    faiss_db.add_documents([embedding])

# Export the vector embeddings database
logging.info("Exporting the vector embeddings database...")
faiss_db.save_local("ipc_embed_db")

# Log completion
logging.info("Process completed successfully.")

# Shutdown Ray after the process
ray.shutdown()
