import streamlit as st
import os
from huggingface_hub import login
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever

st.set_page_config(page_title="AI Chatbot", layout="wide")
st.title("Queries Regarding the File")

st.sidebar.title("Artificial Bot")
st.sidebar.info("Configure and interact with the AI chatbot.")

os.environ["API_KEY"] = "YOUR_API_KEY" #Replace with your huggingface api key
HF_TOKEN = os.getenv("API_KEY")
login(HF_TOKEN)

llm = HuggingFaceInferenceAPI(model_name="mistralai/Mistral-7B-Instruct-v0.3", token=HF_TOKEN)

uploaded_files = st.sidebar.file_uploader("Upload your files:", accept_multiple_files=True)
st.sidebar.subheader("File Information")

if not uploaded_files:
    st.sidebar.warning("Please upload at least one file to proceed.")
    st.stop()

documents = []
for uploaded_file in uploaded_files:
    file_path = os.path.join("./uploaded_files", uploaded_file.name)
    os.makedirs("./uploaded_files", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

reader = SimpleDirectoryReader(input_files=[file_path])
documents.extend(reader.load_data())

st.sidebar.success(f"{len(uploaded_files)} file(s) loaded successfully!")

em = HuggingFaceEmbedding(model_name="all-mpnet-base-v2")
index = VectorStoreIndex.from_documents(documents, embed_model=em)


persist_dir = "./new_db_indexing"
os.makedirs(persist_dir, exist_ok=True)
index.storage_context.persist(persist_dir=persist_dir)

storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
index_new = load_index_from_storage(storage_context, embed_model=em)
retriever = VectorIndexRetriever(index=index_new, similarity_top_k=5)

query_engine = index_new.as_query_engine(llm=llm)
stop_phrases = ['stop', 'exit', 'no']


st.markdown("### 💬 Chat with AI")
query = st.text_input("Ask your question:", placeholder="Type your query here...")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if query and query.lower() not in stop_phrases:
    with st.spinner("Generating response..."):
        response = query_engine.query(query)
        st.success("✅ Response generated!")
        st.session_state["chat_history"].insert(0,{"query": query, "response": response.response})

st.markdown("### 📜 Chat History")
for chat in st.session_state["chat_history"]:
    st.write(f"**Your Query:** {chat['query']}")
    st.markdown(f"**Response:**\n\n{chat['response']}")


