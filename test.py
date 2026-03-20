import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ==========================================
# 1. SETUP THE VECTOR DATABASE (RETRIEVAL)
# ==========================================
print("📚 Loading and indexing document...")

# Load your Vietnamese PDF
pdf_path = "paper.pdf"
loader = PyPDFLoader(pdf_path)
docs = loader.load()

# Split the document into bite-sized paragraphs
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)

# Initialize the multilingual embedding model (Crucial for Vietnamese text!)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Create the local Vector Database
vector_db = Chroma.from_documents(chunks, embeddings)

# Set up the retriever to fetch the top 5 most relevant chunks to avoid missing data
retriever = vector_db.as_retriever(search_kwargs={"k": 5})

# ==========================================
# 2. SETUP THE FAST MERGED MODEL (GENERATION)
# ==========================================
print("⚡ Loading merged model in 4-bit quantization for maximum speed...")

# Point this to your NEW, merged model folder created by the merge script
model_path = "merged_rag_model"

# The compression config for speed and low RAM usage
quantization_config = BitsAndBytesConfig(load_in_4bit=True)

# Load the tokenizer and the single, unified model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    device_map="auto"
)
model.eval()


# ==========================================
# 3. THE RAG PIPELINE FUNCTION
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

    # 1. Prepare the messages exactly like your train.jsonl
    # Note: We use a general RAG system prompt to trigger the logic
    messages = [
        {"role": "system",
         "content": "You are a research assistant. Use the provided evidence to answer the question accurately."},
        {"role": "user", "content": user_question}
    ]

    # Use the tokenizer's built-in template to format the string perfectly
    prompt_stage_1 = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 2. STAGE 1: Generate the Rewrite Tag
    inputs_1 = tokenizer(prompt_stage_1, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs_1 = model.generate(
            **inputs_1,
            max_new_tokens=50,
            temperature=0.1,
            eos_token_id=tokenizer.eos_token_id
        )

    # Extract only the newly generated text (the rewrite part)
    stage_1_gen = tokenizer.decode(outputs_1[0][inputs_1.input_ids.shape[1]:], skip_special_tokens=True)

    # Logic to extract the rewritten query
    search_query = user_question  # Fallback
    if "[S_Rewritten_Query]" in stage_1_gen:
        search_query = stage_1_gen.split("[S_Rewritten_Query]")[-1].split("[")[0].strip()
        print(f"✨ MODEL REWROTE QUERY TO: '{search_query}'")

    # 3. STAGE 2: Retrieval
    print(f"📚 Searching database...")
    retrieved_docs = retriever.invoke(search_query)
    real_evidence = "\n".join([f"Text: {doc.page_content}" for doc in retrieved_docs])

    # 4. STAGE 3: Final Answer
    # We reconstruct the full sequence: [S_Rewritten_Query] -> [R_Evidences] -> [A_Response]
    full_prompt = (
        f"{prompt_stage_1}{stage_1_gen.split('[R_Evidences]')[0].strip()}\n"
        f"[R_Evidences]\n{real_evidence}\n[/R_Evidences]\n"
        f"[A_Response]"
    )

    inputs_2 = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs_2 = model.generate(
            **inputs_2,
            max_new_tokens=150,
            temperature=0.1,
            repetition_penalty=1.15,
            eos_token_id=tokenizer.eos_token_id
        )

    full_gen = tokenizer.decode(outputs_2[0][inputs_2.input_ids.shape[1]:], skip_special_tokens=True)

    # Clean up and return
    final_answer = full_gen.split("[A_Response]")[-1].strip()
    return clean_rag_output(final_answer)


# ==========================================
# 4. CHAT WITH YOUR DATA
# ==========================================
if __name__ == "__main__":
    print("\n✅ System ready! Ask a question about your paper.")

    # A highly specific test question to grab the right context
    my_question = "Mô hình MaMa được huấn luyện trên tập dữ liệu nào"

    answer = ask_rq_rag_chatbot(my_question)

    print("\n🤖 CHATBOT ANSWER:")
    print(answer)