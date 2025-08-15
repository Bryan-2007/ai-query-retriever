import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI

# ==============
# CONFIGURATIONS
# ==============
PDF_PATH = "docs/sample.pdf"
VECTOR_INDEX_PATH = "vector_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_TOKEN = "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"
LLM_MODEL = "openai/gpt-oss-120b:novita"                    # model from HuggingFace LLM

# ================
# STEP 1: LOAD PDF
# ================
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()
print(f"✅ Pages loaded: {len(documents)}")

# =========================
# STEP 2: SPLIT INTO CHUNKS
# =========================
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(documents)
print(f"✅ Total chunks created: {len(chunks)}")

# ======================================
# STEP 3: EMBEDDINGS & FAISS VECTORSTORE
# ======================================
embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vectorstore = FAISS.from_documents(chunks, embedding_model)
vectorstore.save_local(VECTOR_INDEX_PATH)
print(f"✅ FAISS vector store saved to '{VECTOR_INDEX_PATH}/'")

# ======================================
# STEP 4: LOAD VECTORSTORE FOR RETRIEVAL
# ======================================
vector_store = FAISS.load_local(
    VECTOR_INDEX_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# =========================
# STEP 5: GET USER QUESTION
# =========================
user_query = input("\n💬 Ask a question about the document: ")

# ======================
# STEP 6: RETRIEVE RELEVANT CHUNKS
# ======================
docs = retriever.invoke(user_query)
context = "\n\n".join([doc.page_content for doc in docs])
print(f"\n📄 Retrieved {len(docs)} relevant chunks.")

# Show relevant chunks as evidence
print("\n📌 Relevant clauses or evidence:")
for i, doc in enumerate(docs, 1):
    print(f"\n--- Chunk {i} ---\n{doc.page_content}")

# ===============================================
# STEP 7: SEND TO LLM FOR NATURAL LANGUAGE OUTPUT
# ===============================================
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)

completion = client.chat.completions.create(
    model=LLM_MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful assistant that answers based on the provided context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}"}
    ],
)
answer = completion.choices[0].message.content

# ==============================================
# STEP 8: SEND TO LLM FOR STRUCTURED JSON OUTPUT
# ==============================================
json_prompt = f"""
You are an insurance policy evaluator. Based on the given context and question, respond ONLY in valid JSON format with the following keys:
- Insurance Approval: "Approved" or "Rejected"
- payout_amount: A number in INR if mentioned, otherwise "Not specified"

Context:
{context}

Question:
{user_query}
"""
completion_json = client.chat.completions.create(
    model=LLM_MODEL,
    messages=[
        {"role": "system", "content": "Return ONLY valid JSON. No explanations."},
        {"role": "user", "content": json_prompt}
    ],
)
json_output = completion_json.choices[0].message.content

# Checking for possible errors
try:
    parsed_json = json.loads(json_output)
except json.JSONDecodeError:
    parsed_json = {"Insurance Approval": "Error parsing JSON", "payout_amount": "Error"}

# ====================
# STEP 9: SHOW RESULTS
# ====================
print("\n🤖 Answer and 📖 Justification:")
print(answer)

print("\n📊 Final JSON:")
print(json.dumps(parsed_json, indent=4))
