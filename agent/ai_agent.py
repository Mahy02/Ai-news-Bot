# LangChain basics
from langchain.agents import load_tools, Agent
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

# Langchain Loaders:
from langchain.document_loaders import YoutubeLoader
from langchain.document_loaders import WebBaseLoader

# Vector Store and retrievals
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA , RetrievalQAWithSourcesChain

# Chat Prompt templates for dynamic values
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

# htmldate library
from htmldate import find_date

# Youtube
from pytube import YouTube

# From Kor library
from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text, Number


# Supporting libraries
import os
from dotenv import load_dotenv
import requests

# local
from search_tools import YouTubeVideoFinder, SerpSearchTool

class AIAgent(Agent):
    def __init__(self):
        super().__init__()
        self.youtube_tool = YouTubeVideoFinder("What is new about AI today?", 10)
        self.serp_tool = SerpSearchTool("What is new about AI today?")
        self.youtube_urls=[]
        self.website_urls=[]
        self.llm3 = ChatOpenAI(temperature=0,
                  model_name="gpt-3.5-turbo-0613",
                  request_timeout = 180
                )
        self.websitedocs= None
        self.youtubedocs= None
        self.website_topics_found= None
        self.youtube_topics_found= None
        self.website_topics_structured= None
        self.youtube_topics_structured= None
        self.website_data= None
        self.youtube_data= None
        self.website_db: FAISS
        self.youtube_db: FAISS
        self.tweets= dict


    def search_ai_urls(self, input_dict):
        # Use the YouTube search tool to search for videos
        video_urls = self.youtube_tool.find_videos()
        self.youtube_urls= self.youtube_tool.get_youtube_urls()

        # Use the Serp search tool to search for relevant pages
        websites_url_data = self.serp_tool.run(input_dict['query'])  #json

        for i,website in enumerate(websites_url_data):
            self.website_urls.append(website[i]['link'])
        #print(response_data['news'][0]['link'])

    def website_loader(self):
        website_loader = WebBaseLoader(self.website_urls)
        self.website_data = website_loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=2200)
        self.website_docs = text_splitter.split_documents(self.website_data)

        print (f"You have {len(self.website_docs)} docs. First doc is {self.llm3.get_num_tokens(self.website_docs[0].page_content)} tokens")
        print(self.website_docs[0].metadata)

    def transcript_loader(self):
        youtube_loaders = [YoutubeLoader.from_youtube_url(url) for url in self.video_urls]
        transcripts = [loader.load() for loader in youtube_loaders]

        for youtube_loader in youtube_loaders:
            youtube_loader.add_video_info= True
            print(youtube_loader.add_video_info)

        self.youtube_transcript = [doc for sublist in transcripts for doc in sublist]

        text_splitter = RecursiveCharacterTextSplitter(separators=["\n", " "], chunk_size=10000, chunk_overlap=2200)

        self.youtubedocs= text_splitter.split_documents(self.youtube_transcript)

        print (f"You have {len(self.youtubedocs)} docs. First doc is {self.llm3.get_num_tokens(self.youtubedocs[0].page_content)} tokens")


    def create_faiss_vectorstore_websites(self):
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n", " "], chunk_size=4000, chunk_overlap=800)
        docs= text_splitter.split_documents(self.website_data)
        openai_embeddings = OpenAIEmbeddings(show_progress_bar=True, embedding_ctx_length=1024)
        self.website_db = FAISS.from_documents(docs, openai_embeddings)

    def create_faiss_vectorstore_youtube(self):
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n", " "], chunk_size=4000, chunk_overlap=800)
        docs= text_splitter.split_documents(self.youtube_transcript)
        openai_embeddings = OpenAIEmbeddings(show_progress_bar=True, embedding_ctx_length=1024)
        self.youtube_db = FAISS.from_documents(docs, openai_embeddings)


    def extract_websites_topics(self):

        template_1="""
        You are a helpful assistant that helps retrieve distinct topics discussed from many websites' content
        - Your goal is to extract the topic names and brief 1-sentence description of the topic
        - Topics can include:
        - AI tools
        - GPT Models
        - Google Models
        - LLMs
        - llama Models
        - Falcon Models
        - Programming Languages
        - AI recent News
        - AI tutorials
        - OpenAI
        - AI for business 
        - AI for education
        - AI for medicine 
        - AI for art and music
        - Deep Learning
        - NLP
        - Machine Learning
        - Data science
        - Opportunities in AI
        - AI frameworks
        - Future AI
        - Langchain

        - Provide a brief description of the topics after the topic name. Example: 'Topic: Brief Description'
        - Use the same words and terminology that is said in the websites' content
        - Do not respond with numbers, just bullet points of all topics listed under each other. Example:
        Topics:
            - Topic 1 title: topic 1 description
            - Topic 2 title: topic 2 description
            - Topic 3 title: topic 3 description
            
        - Ignore topics on policy and regulations
        - Do not respond with anything outside of the webstes' content. If you can't extract any topics at all in the whole content, say 'Sorry, No topics found in the given content'
        - Only pull topics from the websites' content. Do not use the examples
        - If the authors' names were mentioned in the transcript, instead of saying 'The Author' refer to the names.
        - Make your titles descriptive but concise. Example: 'Shaan's Experience at Twitch' should be 'Shaan's Interesting Projects At Twitch'
        - A topic should be substantial, more than just a one-off comment

        """
        system_message_prompt_map_1 = SystemMessagePromptTemplate.from_template(template_1)

        human_template_1="Websites' Content: {text}" # Simply just pass the text as a human message
        human_message_prompt_map_1 = HumanMessagePromptTemplate.from_template(human_template_1)

        chat_prompt_map = ChatPromptTemplate.from_messages(messages=[system_message_prompt_map_1, human_message_prompt_map_1])


        template_2="""
        You are a helpful assistant that helps retrieve topics discussed in websites' content
        - You will be given a series of bullet topics of topics found
        - Your goal is to exract the topic names and brief 1-sentence description of the topic
        - Do not respond with numbers, just bullet points of all topics listed under each other.
        - Deduplicate any bullet points you see
        - If you think two or more topics are similar and can be merged, merge them together with one topic title and create a new description that fits the merged topics
        - Only pull topics from the websites' content. Do not use the examples.
        """
        system_message_prompt_map_2 = SystemMessagePromptTemplate.from_template(template_2)

        human_template_2="Websites' Content: {text}" # Simply just pass the text as a human message
        human_message_prompt_map_2 = HumanMessagePromptTemplate.from_template(human_template_2)

        chat_prompt_combine = ChatPromptTemplate.from_messages(messages=[system_message_prompt_map_2, human_message_prompt_map_2])


        chain = load_summarize_chain(self.llm3,
                             chain_type="map_reduce",
                             map_prompt=chat_prompt_map,
                             combine_prompt=chat_prompt_combine,
                            verbose=True
                            )
        self.topics_found = chain.run({"input_documents": self.website_docs})
        


    def extract_youtube_topics(self):    

        template_1="""
        You are a helpful assistant that helps retrieve distinct topics talked about in youtube videos transcripts
        - Your goal is to extract the topic names,  brief 1-sentence description of the topic
        - Topics can include:
        - AI tools
        - GPT Models
        - Google Models
        - LLMs
        - llama Models
        - Falcon Models
        - Programming Languages
        - AI recent News
        - AI tutorials
        - OpenAI
        - AI for business 
        - AI for education
        - AI for medicine 
        - AI for art and music
        - Deep Learning
        - NLP
        - Machine Learning
        - Data science
        - Opportunities in AI
        - AI frameworks
        - Future AI
        - Langchain

        - Provide a brief description of the topics after the topic name, then Provide the source of the topic. Example: 'Topic: Brief Description'
        - Use the same words and terminology that is said in the youtube video
        - ALWAYS include Brief Description beside the Topic. Example: 'Topic: Brief Description'
        - Do not respond with numbers, just bullet points of all topics listed under each other. Example:
        Topics:
            - Topic 1 title: topic 1 description
            - Topic 2 title: topic 2 description 
            - Topic 3 title: topic 3 description
            
        - Ignore topics on policy and regulations
        - Do not respond with anything outside of the transcript. If you can't extract any topics at all in the whole transcript, say 'Sorry, No topics found in the given content'
        - Only pull topics from the transcript. Do not use the examples
        - If the speakers names were mentioned in the transcript, instead of saying 'The speaker' refer to the names.
        - Make your titles descriptive but concise. Example: 'Shaan's Experience at Twitch' should be 'Shaan's Interesting Projects At Twitch'
        - A topic should be substantial, more than just a one-off comment

        """
        #- Do not respond with anything outside of the transcript. If you don't see any topics, say, 'No Topics'
        system_message_prompt_map_1 = SystemMessagePromptTemplate.from_template(template_1)

        human_template_1="Transcript: {text}" # Simply just pass the text as a human message
        human_message_prompt_map_1 = HumanMessagePromptTemplate.from_template(human_template_1)

        chat_prompt_map = ChatPromptTemplate.from_messages(messages=[system_message_prompt_map_1, human_message_prompt_map_1])


        template_2="""
        You are a helpful assistant that helps retrieve topics talked about in a youtube transcript
        - You will be given a series of bullet topics of topics found
        - Your goal is to exract the topic names and brief 1-sentence description of the topic
        - Do not respond with numbers, just bullet points of all topics listed under each other.
        - Deduplicate any bullet points you see
        - If you think two or more topics are similar and can be merged, merge them together with one topic title and create a new description that fits the merged topics
        - Only pull topics from the transcript. Do not use the examples.
        """
        system_message_prompt_map_2 = SystemMessagePromptTemplate.from_template(template_2)

        human_template_2="Transcript: {text}" # Simply just pass the text as a human message
        human_message_prompt_map_2 = HumanMessagePromptTemplate.from_template(human_template_2)

        chat_prompt_combine = ChatPromptTemplate.from_messages(messages=[system_message_prompt_map_2, human_message_prompt_map_2])

        chain = load_summarize_chain(self.llm3,
                             chain_type="map_reduce",
                             map_prompt=chat_prompt_map,
                             combine_prompt=chat_prompt_combine,
                            verbose=True
                            )
        
        self.youtube_topics_found = chain.run({"input_documents": self.youtubedocs})

    def structure_website_topics(self):
        schema = Object(
            id="topic",
            description="Topic Information",
            examples=[
                ("Artificial intelligence: MIT News covers advancements, applications, and research in AI technology and algorithms.", [{"title": "Artificial intelligence"}, {"description": "MIT News covers advancements, applications, and research in AI technology and algorithms."}, {"tag": "AI News"}]),
                ("Generative AI: An exciting tool in AI that can generate text based on input prompts.", [{"title": "Generative AI"}, {"description": "An exciting tool in AI that can generate text based on input prompts."}, {"tag": "AI Tools"}]),
                ("AI in relationships: The use of AI in relationship coaching and mentoring, such as the example of an AI-powered romantic relationship coaching app.", [{"title": "AI in relationships"}, {"description": "The use of AI in relationship coaching and mentoring, such as the example of an AI-powered romantic relationship coaching app."}, {"tag": "AI Applications"}]),
                ("AI in different industries: Exploring the potential of AI in various industries beyond consumer software and the internet.", [{"title": "AI in different industries"}, {"description": "Exploring the potential of AI in various industries beyond consumer software and the internet."}, {"tag": "AI Opportunities"}]),
                ("Supervised Learning: A technique in AI that is good at labeling things or computing input to outputs.", [{"title": "Supervised Learning"}, {"description": "A technique in AI that is good at labeling things or computing input to outputs."}, {"tag": "AI Tools"}]),
                ("Large Language Models: The power and potential of large language models in AI applications.", [{"title": "Large Language Models"}, {"description": "The power and potential of large language models in AI applications."}, {"tag": "AI LLMs"}]),
                ("Future growth of AI technologies: The prediction that supervised learning and generative AI will continue to grow in value and adoption over the next three years, with the potential for even greater expansion in the long term.", [{"title": "Future growth of AI technologies:"}, {"description": "The prediction that supervised learning and generative AI will continue to grow in value and adoption over the next three years, with the potential for even greater expansion in the long term."}, {"tag": "Future of AI"}]),
                ("Large Language Models: Dr. Andrew explains how large language models, like GPT, are built using supervised learning to predict the next word, enabling applications that can be built faster and more efficiently.", [{"title": "Large Language Models:"}, {"description": " Dr. Andrew explains how large language models, like GPT, are built using supervised learning to predict the next word, enabling applications that can be built faster and more efficiently."}, {"tag": "AI LLMs"}]),
        
            ],
            attributes=[
                Text(
                    id="title",
                    description="The title of the topic listed",
                ),
                Text(
                    id="description",
                    description="The description of the topic listed",
                ),
                Text(
                    id="tag",
                    description="The type of content being described",
                )
            ],
            many=True,
        )
        chain = create_extraction_chain(self.llm3, schema)
        website_topics_structured_m = chain.run(self.website_topics_found)
        self.website_topics_structured= website_topics_structured_m["data"]["topic"]


    def structure_youtube_topics(self):
        schema = Object(
            id="topic",
            description="Topic Information",
            examples=[
                ("Generative AI: An exciting tool in AI that can generate text based on input prompts.", [{"topic title": "Generative AI"}, {"description": "An exciting tool in AI that can generate text based on input prompts."}, {"tag": "AI Tools"}]),
                ("AI in relationships: The use of AI in relationship coaching and mentoring, such as the example of an AI-powered romantic relationship coaching app.", [{"topic title": "AI in relationships"}, {"description": "The use of AI in relationship coaching and mentoring, such as the example of an AI-powered romantic relationship coaching app."}, {"tag": "AI Applications"}]),
                ("AI in different industries: Exploring the potential of AI in various industries beyond consumer software and the internet.", [{"topic title": "AI in different industries"}, {"description": "Exploring the potential of AI in various industries beyond consumer software and the internet."}, {"tag": "AI Opportunities"}]),
                ("Supervised Learning: A technique in AI that is good at labeling things or computing input to outputs.", [{"topic title": "Supervised Learning"}, {"description": "A technique in AI that is good at labeling things or computing input to outputs."}, {"tag": "AI Tools"}]),
                ("Large Language Models: The power and potential of large language models in AI applications.", [{"topic title": "Large Language Models"}, {"description": "The power and potential of large language models in AI applications."}, {"tag": "AI LLMs"}]),
                ("Future growth of AI technologies: The prediction that supervised learning and generative AI will continue to grow in value and adoption over the next three years, with the potential for even greater expansion in the long term.", [{"topic title": "Future growth of AI technologies:"}, {"description": "The prediction that supervised learning and generative AI will continue to grow in value and adoption over the next three years, with the potential for even greater expansion in the long term."}, {"tag": "Future of AI"}]),
                ("Large Language Models: Dr. Andrew explains how large language models, like GPT, are built using supervised learning to predict the next word, enabling applications that can be built faster and more efficiently.", [{"topic title": "Large Language Models:"}, {"description": " Dr. Andrew explains how large language models, like GPT, are built using supervised learning to predict the next word, enabling applications that can be built faster and more efficiently."}, {"tag": "AI LLMs"}]),
        
            ],
            attributes=[
                Text(
                    id="title",
                    description="The title of the topic listed",
                ),
                Text(
                    id="description",
                    description="The description of the topic listed",
                ),
                Text(
                    id="tag",
                    description="The type of content being described",
                )
            ],
            many=True,
        )
        chain = create_extraction_chain(self.llm3, schema)
        youtube_topics_structured_m = chain.run(self.youtube_topics_found)
        self.youtube_topics_structured= youtube_topics_structured_m["data"]["topic"]
        

    def summarize_youtube_topics(self):
        system_template = """
        You will be given text from youtube transcripts which contains many topics.
        You goal is to write a summary (5 sentences or less) about a topic the user chooses
        The summary should be relative to the description assossiated with the Topic Description
        Do not respond with information that isn't relevant to the topic that the user gives you
        ----------------
        {context}"""

        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]

        # This will pull the two messages together and get them ready to be sent to the LLM through the retriever
        CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)

        
        # I'm also setting k=4 so the number of relevant docs we get back is 4. 
        qa = RetrievalQA.from_chain_type(llm=self.llm3,
                                        chain_type="stuff",
                                        retriever=self.youtube_db.as_retriever(k=4),
                                        chain_type_kwargs = {
                                            'verbose': True,
                                            'prompt': CHAT_PROMPT
                                        },
                                        return_source_documents=True)
        
        

        for topic in self.youtube_topics_structured:
            query = f"""
                {topic['title']}: {topic['description']}
            """
            #expanded_topic = qa.run(query)
            expanded_topic = qa({"query": query})
            print(f"{topic['title']}: {topic['description']}")
            print ("\n\n")
            print(expanded_topic['result'])
            print ("\n\n")
            print(type(expanded_topic['source_documents'][0]))  # document
            print(type(expanded_topic['source_documents']))  #list
            print ("\n\n")

            for topic in expanded_topic['source_documents']:
                print(type(topic)) #document
                print(topic.metadata['source'])

        print(expanded_topic['source_documents'][0].metadata['source'])

        youtube_video_ID=expanded_topic['source_documents'][0].metadata['source']
        youtube_video_URL=f"https://www.youtube.com/watch?v={youtube_video_ID}"
        print(youtube_video_URL)

        #docs[0].metadata['source']
                                        

    def summarize_website_topics(self):
        
        system_template = """
        You will be given text from websites content which contains many topics.
        You goal is to write a summary (5 sentences or less) about a topic the user chooses
        The summary should be relative to the description assossiated with the Topic Description
        Do not respond with information that isn't relevant to the topic that the user gives you
        ----------------
        {context}"""

        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]


        # This will pull the two messages together and get them ready to be sent to the LLM through the retriever
        CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)
   
        # I'm also setting k=4 so the number of relevant docs we get back is 4. 
        qa = RetrievalQA.from_chain_type(llm=self.llm3,
                                        chain_type="stuff",
                                        retriever=self.website_db.as_retriever(k=4),
                                        chain_type_kwargs = {
                                            'verbose': True,
                                            'prompt': CHAT_PROMPT
                                        },
                                        return_source_documents=True)
                                 

        for topic in self.website_topics_structured:
            query = f"""
                {topic['title']}: {topic['description']}
            """
            #expanded_topic = qa.run(query)
            expanded_topic = qa({"query": query})
            print(f"{topic['title']}: {topic['description']}")
            print ("\n\n")
            print(expanded_topic['result'])
            print ("\n\n")
            print(expanded_topic['source_documents'])
            print ("\n\n")


    # def get_tweet_title(self):
    #     pass

    # def get_tweet_body(self):
    #     pass

    # def get_tweet_source(self):
    #     pass

    # def get_tweet_date(self):
    #     pass


    def get_website_publish_date(self, website_link) -> str:
        html=requests.get(website_link).content.decode('utf-8')
        date = find_date(html)
        return date
    
    def get_youtube_info(self, youtube_link) -> dict:
        yt = YouTube(youtube_link)
        print(yt.author)
        video_info = {
                    "publish_date": yt.publish_date.strftime("%Y-%m-%d %H:%M:%S")
                    if yt.publish_date
                    else "Unknown",
                    "watch_url": yt.watch_url,
                    "author": yt.author or "Unknown",
                }
        return video_info



    def get_topic_timestamp_in_video(self):
        system_template = """
        What is the first timestamp when the speakers started talking about a topic the user gives?
        Only respond with the timestamp, nothing else. Example: 0:18:24
        ----------------
        {context}"""
        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
        CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)

        qa = RetrievalQA.from_chain_type(llm=self.llm3,
                                 chain_type="stuff",
                                 retriever=self.youtube_db.as_retriever(k=4),
                                 chain_type_kwargs = {
#                                      'verbose': True,
                                     'prompt': CHAT_PROMPT
                                 })
        
        # Holder for our topic timestamps
        topic_timestamps = []

        for topic in self.youtube_topics_structured:

            query = f"{topic['title']} - {topic['description']}"
            timestamp = qa.run(query)
            
            topic_timestamps.append(f"{topic['title']}: {timestamp}.")

        topic_timestamps_n="\n".join(sorted(topic_timestamps))
        schema_time = Object(
            id="topic",
            description="Topic Information",
            examples=[
                ("Generative AI: 0:18:24 ", [{"topic title": "Generative AI"}, {"timestamp": "0:18:24"}]),
                ("AI in relationships: 0:17:24 ", [{"topic title": "AI in relationships"},{"timestamp": "0:17:24"}]),
                ("AI in different industries: 0:10:24", [{"topic title": "AI in different industries"},  {"timestamp": "0:10:24"}]),
                ("Supervised Learning: 0:12:24 ", [{"topic title": "Supervised Learning"}, {"timestamp":"0:12:24"}]),
                ("Large Language Models: 0:09:00", [{"topic title": "Large Language Models"},  {"timestamp":"0:09:00"}]),
                ("Future growth of AI technologies: 0:05:02", [{"topic title": "Future growth of AI technologies:"}, {"timestamp":"0:05:02"}]),
                ("Large Language Models: 0:02:10", [{"topic title": "Large Language Models:"}, {"timestamp":"0:02:10"}]),
        
            ],
            attributes=[
                Text(
                    id="title",
                    description="The title of the topic listed",
                ),
                Text(
                    id="timestamp",
                    description="The timestamp of topic listed",
                )
            ],
            many=True,
        )
        chain_with_time = create_extraction_chain(self.llm3, schema_time)
        topics_structured_with_time= chain_with_time.run(topic_timestamps_n)