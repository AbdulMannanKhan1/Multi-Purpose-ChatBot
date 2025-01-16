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
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import math
from gtts import gTTS
import tempfile
# from langchain_groq import ChatGroq