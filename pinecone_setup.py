from pinecone import Pinecone, ServerlessSpec
import streamlit as st
from api import PINECONE_API_KEY, PINECONE_ENV

# Initialize Pinecone
index_name = "rag01"
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

def setup():
    # Ensure the Pinecone index exists
    try:
        index = pc.Index(index_name)
        return index
    except Exception:
        st.warning(f"Index '{index_name}' not found. Creating a new index.")
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        index = pc.Index(index_name)
        return index