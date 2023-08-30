#Here we are going to load docs, links..etc and save it to our db vectorstore

from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface_hub import HuggingFaceHubEmbeddings
#from langchain.llms import HuggingFaceHub
from langchain import LLMChain, HuggingFaceHub
from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
from langchain import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import LlamaCppEmbeddings
import textwrap


load_dotenv(find_dotenv())

#embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#llm= HuggingFaceHub(repo_id="Amitesh007/text_generation-finetuned-gpt2")


model_path="models/llama-2-7b-chat.gguf.q2_K.bin"

print("Loading model...")

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# template = """Question: {question}

# Answer: Let's work this out in a step by step way to be sure we have the right answer."""

# prompt = PromptTemplate(template=template, input_variables=["question"])

llm = LlamaCpp(
    model_path=model_path,
    temperature=0.75,
    max_tokens=2000,
    top_p=1,
    callback_manager=callback_manager, 
    verbose=True, # Verbose is required to pass to the callback manager
)
print("Model Loaded & Computer didn't crash, congrats! :) ")

#llama_embeddings = LlamaCppEmbeddings(model_path=model_path, verbose=True,  n_ctx=2048)
miniLM_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
hf_embeddings = HuggingFaceEmbeddings()
print("embeddings loaded")

#####

def create_db_from_youtube_video_url(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)
  
    db = FAISS.from_documents(docs, hf_embeddings)
    print("here after db")
    return db

#This is just for testing the model and the youtube videos
def get_response_from_query(db, query, k=4):

    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])


    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript.
        
        Answer the following question: {question}
        By searching the following video transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs


# Example usage:
video_url = "https://www.youtube.com/watch?v=L_Guz73e6fw"
db = create_db_from_youtube_video_url(video_url)

print("db created")

query = "What are they saying about Microsoft?"

print(query)
response, docs = get_response_from_query(db, query)
print("response")
print(textwrap.fill(response, width=85))




