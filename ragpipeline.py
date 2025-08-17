import os
import fitz
from pinecone import Pinecone , ServerlessSpec
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()



PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

print("Pinecone API Key:", PINECONE_API_KEY)

genai.configure(api_key = GEMINI_API_KEY)
model=genai.GenerativeModel("gemini-1.5-flash")

pc = Pinecone(api_key = PINECONE_API_KEY)
Index_Name = "pdf-chatbot-index"

if Index_Name not in [i['name'] for i in pc.list_indexes()]:
    pc.create_index(
        name=Index_Name,
        dimension=768,   # match your embedding model
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(Index_Name)

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text")
        return text    

def chunk_text(text,chunk_size = 500):
    words = text.split()
    for i in range(0,len(words),chunk_size):
        yield " ".join(words[i:i+chunk_size])    

# def embed_text(text):
#     embedding = model.embed_content(
#         model = "models/embedding-001",
#         content = text
#     )
#     return embedding.embedding

def embed_text(text):
    embedding = genai.embed_content(
        model="models/embedding-001",
        content=text
    )
    return embedding["embedding"]

def process_pdf_and_store(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    for i,chunk in enumerate(chunk_text(text)):
        vector = embed_text(chunk)
        index.upsert([(f"{pdf_path}_{i}",vector,{"text":chunk})])

def query_rag(query):
    query_vector = embed_text(query)
    results = index.query(vector = query_vector, top_k = 5 , include_metadata = True)
    context = "\n".join([match["metadata"]["text"] for match in results["matches"]]) 
    prompt = f"Answer the question based only on the context below:\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:" 
    response = model.generate_content(prompt)
    return response.text.strip()      