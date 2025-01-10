import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.document_loaders import SeleniumURLLoader
import pandas as pd
import PyPDF2
from langchain_google_genai import ChatGoogleGenerativeAI
import math
from gtts import gTTS
import tempfile

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize model and embeddings
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load the Whisper processor and model
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

# Initialize Pinecone
index_name = "rag01"
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Ensure the Pinecone index exists
try:
    index = pc.Index(index_name)
except Exception:
    st.warning(f"Index '{index_name}' not found. Creating a new index.")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    index = pc.Index(index_name)

vector_store = PineconeVectorStore(embedding=embeddings, index=index)
retriever = vector_store.as_retriever()

# Streamlit app
st.title("Multi-Purpose RAG Chatbot")

# Functions for file processing
def process_csv(file):
    df = pd.read_csv(file)
    return df.to_string()

def process_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text_data = ""
    for page in reader.pages:
        text_data += page.extract_text()
    return text_data

def process_text_file(file):
    return file.getvalue().decode("utf-8")

def process_url(url):
    loader = SeleniumURLLoader(urls=[url])
    data = loader.load()
    return data[0].page_content if data else None

def process_audio(file):
    audio_array, sampling_rate = librosa.load(file, sr=16000)
    audio_array, _ = librosa.effects.trim(audio_array)
    chunk_size = 30 * 16000  # 30 seconds
    num_chunks = math.ceil(len(audio_array) / chunk_size)
    text_data = ""
    for i in range(num_chunks):
        chunk = audio_array[i * chunk_size : (i + 1) * chunk_size]
        input_features = processor(chunk, sampling_rate=sampling_rate, return_tensors="pt").input_features
        predicted_ids = model.generate(input_features, max_length=1500,
                                       forced_decoder_ids=processor.get_decoder_prompt_ids(language="en", task="transcribe"))
        text_data += processor.batch_decode(predicted_ids, skip_special_tokens=True)[0] + " "
    return text_data

# Handle state reset on new data
if "current_data_type" not in st.session_state:
    st.session_state.current_data_type = None
if "text_data" not in st.session_state:
    st.session_state.text_data = ""
if "embeddings_created" not in st.session_state:
    st.session_state.embeddings_created = False

option = st.selectbox("How would you like to provide your data?", 
                      ("Select an option", "CSV", "PDF", "Text file", "URL", "Audio"))

uploaded_data = None
data_name = None

if option != "Select an option":
    if option == "CSV":
        uploaded_data = st.file_uploader("Choose a CSV file", type=["csv"])
        data_name = "uploaded_csv"
    elif option == "PDF":
        uploaded_data = st.file_uploader("Choose a PDF file", type=["pdf"])
        data_name = "uploaded_pdf"
    elif option == "Text file":
        uploaded_data = st.file_uploader("Choose a Text file", type=["txt"])
        data_name = "uploaded_text"
    elif option == "URL":
        uploaded_data = st.text_input("Enter URL")
        data_name = "entered_url"
    elif option == "Audio":
        uploaded_data = st.file_uploader("Upload an audio file (.mp3, .wav)", type=["mp3", "wav"])
        data_name = "uploaded_audio"

if data_name != st.session_state.current_data_type:
    st.session_state.text_data = ""
    st.session_state.embeddings_created = False
    st.session_state.current_data_type = data_name
    if "text_data" in st.session_state:
        index.delete(delete_all=True)  # Reset the Pinecone index

# Processing data
if uploaded_data and not st.session_state.embeddings_created:
    try:
        with st.spinner('Processing data and creating embeddings...'):
            if data_name == "uploaded_csv":
                st.session_state.text_data = process_csv(uploaded_data)
            elif data_name == "uploaded_pdf":
                st.session_state.text_data = process_pdf(uploaded_data)
            elif data_name == "uploaded_text":
                st.session_state.text_data = process_text_file(uploaded_data)
            elif data_name == "entered_url":
                st.session_state.text_data = process_url(uploaded_data)
                if not st.session_state.text_data:
                    st.error("Failed to load URL content.")
                    st.stop()
            elif data_name == "uploaded_audio":
                st.session_state.text_data = process_audio(uploaded_data)
            
            # Creating embeddings
            documents = [Document(page_content=st.session_state.text_data)]
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            docs = text_splitter.split_documents(documents)
            for doc in tqdm(docs):
                metadata = {"text": doc.page_content}
                vector = embeddings.embed_query(doc.page_content)
                doc_id = str(hash(doc.page_content))
                index.upsert(vectors=[(doc_id, vector, metadata)])
            st.session_state.embeddings_created = True
            st.success("Embeddings created successfully!")
    except Exception as e:
        st.error(f"Error: {e}")

# Query handling
if st.session_state.embeddings_created:
    query = st.text_input("Enter your query:")
    if query:
        with st.spinner('Retrieving response...'):
            if "text_data" in st.session_state and st.session_state.text_data:
                qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="map_reduce",
                                                       retriever=retriever)
                response = qa_chain.invoke(query)
                # Display the response text
                st.subheader("Result:")
                st.write(response["result"])

                try:
                    # Generate TTS audio using gTTS
                    tts = gTTS(response["result"])
                    
                    # Create a temporary file to store the audio
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
                        temp_path = tmpfile.name
                        tts.save(temp_path)

                    # Play audio in Streamlit
                    st.audio(temp_path, format="audio/mp3")

                    # Optionally clean up the temporary file
                    os.remove(temp_path)

                except Exception as e:
                    st.error(f"Error during TTS conversion: {e}")
            else:
                st.error("No context available for querying.")