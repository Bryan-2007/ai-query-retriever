import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# === Step 1: Load PDF ===
pdf_path = "docs/sample.pdf"  # Update path if needed
loader = PyPDFLoader(pdf_path)
documents = loader.load()

print(f"✅ Pages loaded: {len(documents)}")
print("\n--- First Page Content ---")
print(documents[0].page_content[:500])

# === Step 2: Split into Chunks ===
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(documents)

print(f"\n✅ Total chunks created: {len(chunks)}")
for i, chunk in enumerate(chunks[:2]):
    print(f"\n--- Chunk {i+1} ---\n{chunk.page_content[:300]}")

# === Step 3: Create Embeddings ===
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === Step 4: Create and Save FAISS Vector Store ===
vectorstore = FAISS.from_documents(chunks, embedding_model)
vectorstore.save_local("vector_index")
print("\n✅ FAISS vector store created and saved to 'vector_index/'")

# === Step 5: Load Vector Store and Query ===
vector_store = FAISS.load_local(
    "vector_index",
    embedding_model,
    allow_dangerous_deserialization=True
)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

query = "What precautions should I take before knee replacement surgery?"
docs = retriever.invoke(query)

print("\n--- Retrieved Chunks ---")
for i, doc in enumerate(docs):
    print(f"\n>> Chunk {i+1}:\n{doc.page_content}")