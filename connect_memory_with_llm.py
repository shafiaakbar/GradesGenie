import os
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# Step 1: Setup LLM Mistral w HuggingFace
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGING_FACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"


def load_llm(huggingface_repo_id):
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        hf_token=HF_TOKEN,
        model_kwargs={"max_length": 512}
    )

# Step 2: Define Custom Prompt
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
Don't provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk, please.
"""

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# Step 3: Load FAISS Vectorstore
DB_FAISS_PATH = "vectorstore/db_faiss_images"

# Check if FAISS index exists
if not os.path.exists(os.path.join(DB_FAISS_PATH, "index.faiss")):
    raise FileNotFoundError("FAISS index not found! Run the embedding script first.")

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS index with correct embedding model
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGING_FACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Step 4: Take User Query and Invoke Model
user_query = input("Write your Query Here: ")
response = qa_chain.invoke({'query': user_query})  # 
# Display results
print("Result:", response["result"])
print("Source Documents:", response["source_documents"])
