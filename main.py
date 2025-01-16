from libraries import *
from functions import *
from pinecone_setup import setup
from langchain_groq import ChatGroq


# Initialize model and embeddings
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
# # llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-002")
# llm = ChatGroq(model="llama-3.3-70b-versatile")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Load the Whisper processor and model
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")


index = setup()
vector_store = PineconeVectorStore(embedding=embeddings, index=index)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})


# Streamlit app
st.title("Multi-Purpose RAG Chatbot")


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
    index = setup()
    if "text_data" in st.session_state:
        # Check if the Pinecone index contains vectors
        index_stats = index.describe_index_stats()
        vector_count = index_stats["total_vector_count"]

        if vector_count > 0:
            index.delete(delete_all=True)  # Reset the Pinecone index if it contains vectors
            st.info(f"Deleted {vector_count} existing vectors in the index.")

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
                st.session_state.text_data = process_audio(uploaded_data,processor,model)
            
            # Creating embeddings
            documents = [Document(page_content=st.session_state.text_data)]
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
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
                
                qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="refine",
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