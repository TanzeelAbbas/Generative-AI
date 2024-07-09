import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from tqdm import tqdm
import time
import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

# Step 1: Load and process documents
def load_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(directory, filename))
            documents.extend(loader.load())
    return documents

# Step 2: Split documents into chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

# Step 3: Create embeddings and vector store
def create_vector_store(chunks):
    embedding_model = HuggingFaceBgeEmbeddings(
                model_name="BAAI/bge-small-en",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
    vector_store = Chroma.from_documents(chunks, embedding_model)
    return vector_store


# Step 4: Set up LLM
def setup_mistral_moe():
    api_key = "PASTE API KEY HERE"
    base_url="BASE URL"        # Examples: http://172.31.0.1:8001/v1, http://localhost:8001/v1 etc
    model_name = "MODEL NAME"  # Examples: llama3, mistral-moe etc
    
    llm = ChatOpenAI(api_key=api_key, base_url = base_url ,model=model_name)
    return llm

# Step 5: Create QA chain
def create_qa_chain(vector_store, llm):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )

# Step 6: Generate questions and answers
def generate_qa_pairs(qa_chain, chunks, questions_per_chunk=2):
    qa_pairs = []
    for chunk in tqdm(chunks, desc="Generating QA pairs"):
        for _ in range(questions_per_chunk):
            try:
                question_query = f"Generate a question based on the following context: {chunk.page_content}"
                question_result = qa_chain({"query": question_query})
                question = question_result['result'].strip()
                
                answer_result = qa_chain({"query": question})
                answer = answer_result['result'].strip()
                
                qa_pairs.append({"question": question, "answer": answer})
                
                # Periodically save results
                if len(qa_pairs) % 100 == 0:
                    with open("qa_pairs_partial.json", "w") as f:
                        json.dump(qa_pairs, f, indent=2)
                
                time.sleep(1)  # To avoid hitting rate limits
            except Exception as e:
                print(f"Error generating QA pair: {e}")
                time.sleep(5)  # Wait a bit longer before retrying
    return qa_pairs


if __name__ == "__main__":
    directory = "../../data/raw/01-text-files"
    
    print("Loading documents...")
    documents = load_documents(directory)
    
    print("Splitting documents...")
    chunks = split_documents(documents)
    
    print("Creating vector store...")
    vector_store = create_vector_store(chunks)
    
    print("Setting up Mistral MoE model...")
    llm = setup_mistral_moe()
    
    print("Creating QA chain...")
    qa_chain = create_qa_chain(vector_store, llm)
    
    print("Generating QA pairs...")
    qa_pairs = generate_qa_pairs(qa_chain, chunks)
    
    print(f"Generated {len(qa_pairs)} QA pairs.")
    
    for i, qa in enumerate(qa_pairs, 1):
        print(f"Q{i}: {qa['question']}")
        print(f"A{i}: {qa['answer']}")
        print()
    
    import json
    with open("../../data/synthetic-finetuning-data/fine_tuning_data", "w") as f:
        json.dump(qa_pairs, f, indent=2)
