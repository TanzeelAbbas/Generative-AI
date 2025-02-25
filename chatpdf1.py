from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain import HuggingFaceHub
import os
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("HUGGINGFACEHUB_API_TOKEN")


def loadSplitDocuments(file_path, chunk_size, chunk_overlap):
  loader = DirectoryLoader(file_path, glob = "*.txt", loader_cls=TextLoader)
  documents = loader.load()
  text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap= chunk_overlap)
  text  = text_splitter.split_documents(documents)
  return text



def load_embedding_model():
    model_name = "BAAI/bge-large-en"
    model_kwargs = {"device": "cuda"}
    encode_kwargs = {"normalize_embeddings": True}

    bgeEmbeddings = HuggingFaceBgeEmbeddings(
                    model_name=model_name,
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs)
    
    return bgeEmbeddings



def store_embeddings(files_path):
    directory = 'db'
    text = loadSplitDocuments(files_path, chunk_size = 600, chunk_overlap=60)
    
    bgeEmbeddings = load_embedding_model()
    vectordb = Chroma.from_documents(
                        documents = text,
                        embedding = bgeEmbeddings,
                        persist_directory= directory )

    vectordb.persist()
 
  
    
def get_retriever(db_directory):
    
    directory = db_directory
    bgeEmbeddings = load_embedding_model()
    
    vectordb = Chroma(persist_directory = directory, embedding_function= bgeEmbeddings)
    return vectordb
    

def get_HFModel(model_name):
    repo_id = model_name
    llm = HuggingFaceHub(
        repo_id=repo_id, model_kwargs={"temperature": 0, "max_length": 64}
    )
    return llm


def get_conversational_chain(model_name):

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer.
    Context: {context}
    Question: {question}
    Chat History: {chat_history}
    Answer:
    """
    prompt = PromptTemplate(template = prompt_template)
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    vectordb = get_retriever()
    llm = get_HFModel(model_name)
    
    chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectordb.as_retriever(search_kwargs = {"k":4}),
                #verbose=True,
                memory=memory,
                condense_question_prompt=prompt,
                rephrase_question=False
    )

    return chain


def main():
    get_conversational_chain("google/flan-t5-xxl")

if __name__ == "__main__":
    store_embeddings("")
    main()
