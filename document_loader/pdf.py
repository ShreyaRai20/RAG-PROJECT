from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings

load_dotenv()

data = PyPDFLoader('document_loader/GRU.pdf')
docs = data.load()
texts = [doc.page_content for doc in docs]

embedding = MistralAIEmbeddings(model="mistral-embed",)

print(f"length of docs: {len(docs)}")

for doc in docs:
    print(doc.page_content)
    
embedded = embedding.embed_documents(texts)

print(embedded)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)

chunks = text_splitter.split_(docs)

    
