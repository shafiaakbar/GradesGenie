from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import pytesseract
from PIL import Image
import os


# Set Tesseract Path (only for Windows users)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
#load html data
DATA_PATH = "data/"

def extract_text_from_images(data_path):
    text_data = []
    
    for file in os.listdir(data_path):
        if file.endswith((".png", ".jpg", ".jpeg")):  # Process only image files
            image_path = os.path.join(data_path, file)
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)  # Extract text using OCR
            text_data.append({"text": text, "source": file})
    
    return text_data

image_texts = extract_text_from_images(DATA_PATH)
print("Extracted Text from Images:", image_texts)

#Create Chunks(break long text into small text)

def create_chunks_from_text(text_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.create_documents([item["text"] for item in text_data])
    return text_chunks

text_chunks = create_chunks_from_text(image_texts)
print((text_chunks))

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Store embeddings in FAISS
DB_FAISS_PATH = "vectorstore/db_faiss_images"
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)





# def extract_text_with_ocr(pdf_path):
#     pages = convert_from_path(pdf_path)  # Convert PDF pages to images
#     text = ""
#     for page in pages:
#         text += pytesseract.image_to_string(page) + "\n"
#     return text

# pdf_text = extract_text_with_ocr("data/transcript.pdf")
# print(pdf_text)  # Verify the extracted text

