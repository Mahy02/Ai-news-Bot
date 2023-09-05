#Here we are going to load docs, links..etc and save it to our db vectorstore

from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings.huggingface_hub import HuggingFaceHubEmbeddings
#from langchain.llms import HuggingFaceHub
from langchain import LLMChain, HuggingFaceHub
from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
from langchain import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import LlamaCppEmbeddings
import textwrap


load_dotenv(find_dotenv())



#This is just for testing the model and the youtube videos
# will later be in LLaMaQuant.py

# model_path="models/llama-2-7b-chat.gguf.q4_1.bin"

# print("Loading model...")

# # Callbacks support token-wise streaming
# callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])


# llm = LlamaCpp(
#     model_path=model_path,
#     temperature=0.75,
#     max_tokens=2000,
#     n_ctx=2048,
#     top_p=1,
#     callback_manager=callback_manager, 
#     verbose=True, # Verbose is required to pass to the callback manager
# )




llm = OpenAI(model_name="text-davinci-003")
# print("Model Loaded & Computer didn't crash, congrats! :) ")

## Embeddings:

openai_embeddings = OpenAIEmbeddings()
# miniLM_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# hf_embeddings = HuggingFaceEmbeddings()
print("embeddings loaded")

#####






def create_vector_db(website_urls: list , video_url: str) -> FAISS:
    website_loader = WebBaseLoader(website_urls)
    website_data = website_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    website_docs = text_splitter.split_documents(website_data)

    youtube_loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = youtube_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    youtube_docs = text_splitter.split_documents(transcript)

    websites_db = FAISS.from_documents(website_docs, openai_embeddings)
    youtube_db = FAISS.from_documents(youtube_docs, openai_embeddings)
    youtube_db.merge_from(websites_db)

    db= youtube_db
    
    return db



def get_response_from_query(db, query, k=4):

    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful lovely smart assistant that that can answer questions about recent AI news
        based on the given content from websites and youtube transcripts.
        
        Answer the following question: {question}
        By searching the following content: {docs} 
        
        Only use the factual information from the content to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.

        You should also include the source of the document in your answer.
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs


# def get_response_from_youtube_query(db, query, k=4):

#     """
#     text-davinci-003 can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
#     the number of tokens to analyze.
#     """

#     docs = db.similarity_search(query, k=k)
#     docs_page_content = " ".join([d.page_content for d in docs])


#     prompt = PromptTemplate(
#         input_variables=["question", "docs"],
#         template="""
#         You are a helpful assistant that that can answer questions about youtube videos 
#         based on the video's transcript.
        
#         Answer the following question: {question}
#         By searching the following video transcript: {docs}
        
#         Only use the factual information from the transcript to answer the question.
        
#         If you feel like you don't have enough information to answer the question, say "I don't know".
        
#         Your answers should be verbose and detailed.
#         """,
#     )

#     chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

#     response = chain.run(question=query, docs=docs_page_content)
#     response = response.replace("\n", "")
#     return response, docs

# def get_response_from_websites_query(db, query, k=4):

#     """
#     text-davinci-003 can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
#     the number of tokens to analyze.
#     """

#     docs = db.similarity_search(query, k=k)
#     docs_page_content = " ".join([d.page_content for d in docs])


#     prompt = PromptTemplate(
#         input_variables=["question", "docs"],
#         template="""
#         You are a helpful assistant that that can answer questions about multiple websites links
#         based on the website's content.
        
#         Answer the following question: {question}
#         By searching the following websites content: {docs}
        
#         Only use the factual information from the websites content to answer the question.
        
#         If you feel like you don't have enough information to answer the question, say "I don't know".
        
#         Your answers should be verbose and detailed.
#         """,
#     )

#     chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

#     response = chain.run(question=query, docs=docs_page_content)
#     response = response.replace("\n", "")
#     return response, docs



# Example usage:


# video_url = "https://www.youtube.com/watch?v=L_Guz73e6fw"
# youtube_db = create_db_from_youtube_video_url(video_url)
# print("youtube db created")
# print(youtube_db)

# websites_db= create_db_from_website_links(["https://www.technologyreview.com/2023/09/05/1078987/the-download-the-climate-tech-companies-to-watch-and-mysterious-ai-models/", "https://www.cnbc.com/2023/09/05/qualcomm-focuses-on-auto-and-ai-as-apple-business-faces-uncertainty.html" , "https://tech.hindustantimes.com/tech/news/5-things-about-ai-you-may-have-missed-today-ai-avatar-at-g20-new-ai-models-in-china-and-more-71693919625316.html"])
# print("website db created")
# print(websites_db)





# youtube_query = "What are they saying about Microsoft?"
# response, docs = get_response_from_youtube_query(youtube_db, youtube_query)
# print(response)
# print(textwrap.fill(response, width=85))


# websites_query = "What is new about AI today?"
# response, docs = get_response_from_websites_query(websites_db, websites_query)
# print(response)

query= "What is new about AI today?"
db= create_vector_db(website_urls= ['https://www.technologyreview.com/2023/09/05/1078987/the-download-the-climate-tech-companies-to-watch-and-mysterious-ai-models/' , 'https://www.cnbc.com/2023/09/05/qualcomm-focuses-on-auto-and-ai-as-apple-business-faces-uncertainty.html' , 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRAqM1YGwaQZtp8ipKL5W9VxwMqzb1CcxN8k0cQ9fo8n3JsYEOBcDQP-ppRxw&s' ] , video_url= 'https://www.youtube.com/watch?v=LK5j3pp0Too&pp=ygUVV2hhdCBpcyBuZXcgYWJvdXQgQUk')
response, docs = get_response_from_query(db, query)
print(response)
print(docs)



