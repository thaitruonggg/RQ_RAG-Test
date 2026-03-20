import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ==========================================
# 1. SETUP THE MULTI-DOC VECTOR DATABASE
# ==========================================
DOCS_PATH = "documents"
DB_PATH = "./chroma_db_storage"

print("📚 Indexing research documents...")

# Create the folder if it doesn't exist
if not os.path.exists(DOCS_PATH):
    os.makedirs(DOCS_PATH)
    print(f"⚠️ Created {DOCS_PATH} folder. Put your PDFs there and restart.")

# Load all PDFs from the directory
loader = DirectoryLoader(DOCS_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)

# Initialize multilingual embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Create/Load persistent Vector Database
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=DB_PATH
)

# Retriever setup (k=5 provides a good balance of context vs noise)
retriever = vector_db.as_retriever(search_kwargs={"k": 7})

# ==========================================
# 2. SETUP THE FAST MERGED MODEL
# ==========================================
print("⚡ Loading merged model in 4-bit...")

model_path = "merged_rag_model"
quantization_config = BitsAndBytesConfig(load_in_4bit=True)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    device_map="auto"
)
model.eval()


# ==========================================
# 3. THE REFINED RAG PIPELINE
# ==========================================
def clean_rag_output(raw_text):
    """Removes looping artifacts and fake Q&A generations."""
    # 1. Chop off anything that starts with "Question:" or "Dựa vào..." repeating
    if "Question:" in raw_text:
        raw_text = raw_text.split("Question:")[0]

    # 2. Split the text by newlines and grab only the first actual paragraph
    lines = raw_text.split("\n")

    # 3. Clean up the first valid line
    for line in lines:
        clean_line = line.strip()
        if len(clean_line) > 5:  # Ignore empty lines or weird short artifacts
            return clean_line

    return raw_text.strip()

def ask_rq_rag_chatbot(user_question):
    print(f"\n👤 USER QUESTION: '{user_question}'")

    # 1. RETRIEVAL (The most important part for accuracy)
    retrieved_docs = retriever.invoke(user_question)

    # Track sources for your display
    source_files = set([os.path.basename(doc.metadata.get('source')) for doc in retrieved_docs])

    # Combine the evidence text clearly
    evidence_text = "\n".join([doc.page_content for doc in retrieved_docs])

    # --- DEBUG: Uncomment this if you still get wrong answers to see what the AI sees ---
    # print(f"DEBUG: Evidence chunk: {evidence_text[:200]}...")

    # 2. THE "PRE-FILLED" PROMPT
    # We use Llama 3 tags but MANUALLY start the assistant's response.
    # This forces the model to skip the 'Helpful Assistant' talk and go straight to RQ-RAG mode.
    print("\n--- DEBUG: WHAT THE DATABASE FOUND ---")
    for i, doc in enumerate(retrieved_docs):
        print(f"Chunk {i + 1}: {doc.page_content[:200]}...")
    print("---------------------------------------\n")

    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"Bạn là một trợ lý nghiên cứu khoa học. Chỉ sử dụng thông tin trong [R_Evidences] để trả lời. "
        f"Nếu không thấy số liệu chính xác trong tài liệu, hãy trả lời 'Tôi không tìm thấy số liệu này'.<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"Câu hỏi: {user_question}\n\n"
        f"[R_Evidences]\n{evidence_text}\n[/R_Evidences]<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"[A_Response]\n"
    )

    # 3. GENERATION
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,  # Keep it short to prevent loops
            temperature=0.1,  # Zero variability for research facts
            repetition_penalty=1.2,  # Stop it from echoing the question
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # 4. DECODE ONLY THE NEW STUFF
    # We only decode what the model added AFTER our pre-filled prompt
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    final_answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Clean and Format
    clean_answer = clean_rag_output(final_answer)
    return f"{clean_answer}\n\n📌 Nguồn: {', '.join(source_files)}"


# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    print("\n✅ System ready! Ask a question about your research library.")

    query = "Yêu cầu hệ điều hành để sử dụng samsung portable ssd là gì?"
    result = ask_rq_rag_chatbot(query)

    print("\n🤖 CHATBOT ANSWER:")
    print(result)