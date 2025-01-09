import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
from tqdm import tqdm
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.document_loaders import SeleniumURLLoader
import pandas as pd
import PyPDF2
from langchain_google_genai import ChatGoogleGenerativeAI
import math
import streamlit as st
import pyttsx3

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY= os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")

# Initialize HuggingFace Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")

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
        dimension=384,  # Dimension change (to match embedding model)
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    index = pc.Index(index_name)

vector_store = PineconeVectorStore(embedding=embeddings, index=index)
retriever = vector_store.as_retriever()  # Initialize retriever

# Streamlit app title
st.title("Multi-Purpose RAG Chatbot")

# Load the processor and model
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

# Initialize the embeddings state
if "embeddings_created" not in st.session_state:
    st.session_state.embeddings_created = False
if "text_data" not in st.session_state:
    st.session_state.text_data = ""

# User input for data source
option = st.selectbox(
    "How would you like to provide your data?",
    ("Select an option", "CSV", "PDF", "Text file", "URL", "Audio"),
)

uploaded_data = None
data_name = None

# Process user file selection
if option != "Select an option":
    if option == "CSV":
        uploaded_data = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded_data:
            data_name = "uploaded_csv"
    elif option == "PDF":
        uploaded_data = st.file_uploader("Choose a PDF file", type=["pdf"])
        if uploaded_data:
            data_name = "uploaded_pdf"
    elif option == "Text file":
        uploaded_data = st.file_uploader("Choose a Text file", type=["txt"])
        if uploaded_data:
            data_name = "uploaded_text"
    elif option == "URL":
        uploaded_data = st.text_input("Enter URL")
        if uploaded_data:
            data_name = "entered_url"
    elif option == "Audio":
        uploaded_data = st.file_uploader("Upload your audio file (.mp3, .wav):",
                                         type=["mp3", "wav"])
        if uploaded_data:
            data_name = "uploaded_audio"

# Processing data and creating embeddings
if uploaded_data and not st.session_state.embeddings_created:
    try:
        with st.spinner('Processing data and creating embeddings...'):
            if data_name == "uploaded_csv":
                df = pd.read_csv(uploaded_data)
                st.session_state.text_data = df.to_string()
            elif data_name == "uploaded_pdf":
                pdf_reader = PyPDF2.PdfReader(uploaded_data)
                st.session_state.text_data = ""
                for page in pdf_reader.pages:
                    st.session_state.text_data += page.extract_text()
            elif data_name == "uploaded_text":
                st.session_state.text_data = uploaded_data.getvalue().decode("utf-8")
            elif data_name == "entered_url":
                loader = SeleniumURLLoader(urls=[uploaded_data])
                data = loader.load()
                st.session_state.text_data = data[0].page_content if data else None
                if not st.session_state.text_data:
                    st.error("Could not load URL content.")
                    st.stop()
            elif data_name == "uploaded_audio":
                with st.spinner("Transcribing audio..."):
                    # Load and process audio
                    audio_array, sampling_rate = librosa.load(uploaded_data, sr=16000)
                    audio_array, _ = librosa.effects.trim(audio_array)  # Trim silence

                    # Chunk parameters
                    chunk_duration = 30  # seconds
                    chunk_size = chunk_duration * 16000
                    num_chunks = math.ceil(len(audio_array) / chunk_size)

                    st.session_state.text_data = ""
                    for i in range(num_chunks):
                        chunk = audio_array[i * chunk_size : (i + 1) * chunk_size]
                        input_features = processor(chunk, sampling_rate=sampling_rate, 
                                                   return_tensors="pt").input_features
                        predicted_ids = model.generate(input_features, max_length=1500, 
                                                       forced_decoder_ids=processor.get_decoder_prompt_ids
                                                       (language="en", task="transcribe"))
                        transcription = processor.batch_decode(predicted_ids, 
                                                               skip_special_tokens=True)[0]
                        st.session_state.text_data += transcription + " "
                
                # Show snippet of the processed data
                # st.write(st.session_state.text_data)
            #st.write(f"Processed text data: {st.session_state.text_data[:5000]}...")

            if st.session_state.text_data:  # Ensure text_data is populated
                documents = [Document(page_content=st.session_state.text_data)]
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                docs = text_splitter.split_documents(documents)

                for doc in tqdm(docs):
                    metadata = {"text": f"{doc.page_content}"}
                    vector = embeddings.embed_query(doc.page_content)
                    doc_id = str(hash(doc.page_content))
                    index.upsert(vectors=[(doc_id, vector, metadata)])

                st.session_state.embeddings_created = True
                st.success("Embeddings created successfully!")
            else:
                st.error("No text data was found or processed.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        
# Query handling if embeddings are created

if st.session_state.embeddings_created:
    query = st.text_input("Enter your query:")
    if query:
        with st.spinner('Getting response...'):
            if "text_data" in st.session_state and st.session_state.text_data:
                qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="map_reduce",
                                                       retriever=retriever)
                response = qa_chain.invoke(query)
                # Display the response text
                st.subheader("Result:")
                st.write(response["result"])

                try:
                    # Initialize the pyttsx3 engine
                    engine = pyttsx3.init()

                    # Get the list of available voices
                    voices = engine.getProperty('voices')

                    # Set the voice to a female/male voice (this may vary depending on your system)
                    engine.setProperty('voice', voices[1].id)  # Change the index to select a different voice

                    # Save the speech audio to a file (response_audio.mp3)
                    audio_file_path = "response_audio.mp3"
                    engine.save_to_file(response, audio_file_path)  # Save speech to mp3
                    engine.runAndWait()
                    st.audio(audio_file_path, format="audio/mp3")
                    os.remove(audio_file_path)

                except Exception as e:
                    print(f"An error occurred: {e}")

                except Exception as e:
                    st.error(f"An error occurred while generating the audio: {e}")
            else:
                st.error("No context available for querying.")