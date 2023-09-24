# from langchain
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage

# from pydantic
from pydantic import BaseModel


# from fastapi
from fastapi import FastAPI

# Supporting libraries
import os
import requests
import json
from dotenv import load_dotenv, find_dotenv

#local
from agent.ai_agent import AIAgent
from twitter.twitter_agent import TwitterHandler



load_dotenv(find_dotenv())


template = """
    You are a very experienced ghostwriter who excels at writing Twitter threads.
You will be given a bunch of info below and a topic headline, your job is to use this info and your own knowledge
to write an engaging Twitter thread.
The first tweet in the thread should have a hook and engage with the user to read on.

Here is your style guide for how to write the thread:
1. Voice and Tone:
Informative and Clear: Prioritize clarity and precision in presenting data. Phrases like "Research indicates," "Studies have shown," and "Experts suggest" impart a tone of credibility.
Casual and Engaging: Maintain a conversational tone using contractions and approachable language. Pose occasional questions to the reader to ensure engagement.
2. Mood:
Educational: Create an atmosphere where the reader feels they're gaining valuable insights or learning something new.
Inviting: Use language that encourages readers to dive deeper, explore more, or engage in a dialogue.
3. Sentence Structure:
Varied Sentence Lengths: Use a mix of succinct points for emphasis and longer explanatory sentences for detail.
Descriptive Sentences: Instead of directive sentences, use descriptive ones to provide information. E.g., "Choosing a topic can lead to..."
4. Transition Style:
Sequential and Logical: Guide the reader through information or steps in a clear, logical sequence.
Visual Emojis: Emojis can still be used as visual cues
5. Rhythm and Pacing:
Steady Flow: Ensure a smooth flow of information, transitioning seamlessly from one point to the next.
Data and Sources: Introduce occasional statistics, study findings, or expert opinions to bolster claims, and offer links or references for deeper dives.
6. Signature Styles:
Intriguing Introductions: Start tweets or threads with a captivating fact, question, or statement to grab attention.
Question and Clarification Format: Begin with a general question or statement and follow up with clarifying information. E.g., "Why is sleep crucial? A study from XYZ University points out..."

Engaging Summaries: Conclude with a concise recap or an invitation for further discussion to keep the conversation going.
Distinctive Indicators for an Informational Twitter Style:

Leading with Facts and Data: Ground the content in researched information, making it credible and valuable.
Engaging Elements: The consistent use of questions and clear, descriptive sentences ensures engagement without leaning heavily on personal anecdotes.
Visual Emojis as Indicators: Emojis are not just for casual conversations; they can be effectively used to mark transitions or emphasize points even in an informational context.
Open-ended Conclusions: Ending with questions or prompts for discussion can engage readers and foster a sense of community around the content.

Last instructions:
The twitter thread should be between the length of 3 and 10 tweets 
Each tweet should start with (tweetnumber/total length)
Dont overuse hashtags, only one or two for entire thread.
The first tweet, do not place a number at the start.
When numbering the tweetes Only the tweetnumber out of the total tweets. i.e. (1/9) not (tweet 1/9)
Only return the thread, no other text, and make each tweet its own paragraph.
Make sure each tweet is lower that 180 chars
    Topic Headline:{topic}
    Info: {info}
    """

#Use links sparingly and only when really needed, but when you do make sure you actually include them AND ONLY PUT THE LINk, dont put brackets around them. 

prompt = PromptTemplate(
    input_variables=["info","topic"], template=template
)

llm3 = ChatOpenAI(temperature=0,
                model_name="gpt-3.5-turbo-0613",
                request_timeout = 180
                )
llm_chain = LLMChain(
    llm=llm3,
    prompt=prompt,
    verbose=True,
)


#app = FastAPI()


# @app.post("/")
def tweeterAgent(tweet_details):
    topic_summary= tweet_details['topic_summary']
    topic_headline= tweet_details['topic_title']
    topic_source= tweet_details['topic_source']
    topic_publishdate= tweet_details['topic_publish_date']
    if "youtube" in topic_source:
        topic_timestamp= tweet_details['topic_timestamp']
    else:
        topic_timestamp= -1

    thread = llm_chain.predict(info = topic_summary, topic = topic_headline)
    #we should also add source, publish date and timstamp here at the end of the thread and pass it to tweetertweet()
    twitter_handler = TwitterHandler()
    ret = twitter_handler.tweetertweet(thread=thread, llm3=llm3, source=topic_source, publish_date=topic_publishdate, timestamp= topic_timestamp)
    print(ret)
    return ret

def main(): 

    ai_agent = AIAgent()

    # 1. Search

    ai_agent.search_ai_news_urls()   #comment for now to not use up much requests

    # - here we can filter out the publish date and make new list with new ones only

    # 2. Load & Split

    ai_agent.website_loader()
    ai_agent.transcript_loader()

    # 3. Extract topics

    ai_agent.extract_websites_topics()
    print(ai_agent._website_topics_found)
    ai_agent.extract_youtube_topics()
    print(ai_agent._youtube_topics_found)

    # 4. Structure topics

    ai_agent.structure_youtube_topics()
    print(ai_agent._youtube_topics_structured)
    ai_agent.structure_website_topics()
    print(ai_agent._website_topics_structured)
    


    # 5. Vector databases FAISS
    ai_agent.create_faiss_vectorstore_websites()
    ai_agent.create_faiss_vectorstore_youtube()


    # 6. Summarize
    ai_agent.summarize_website_topics()
    ai_agent.summarize_youtube_topics()


    # 7. Get Tweets
    tweets: list
    tweets=[
        #{'topic_title': 'Google Models', 'topic_summary': 'Google has recently unveiled new features that utilize AI as a collaborator. These features include upgrades to Google Maps, real-time weather and traffic conditions presented in a 3D world, and a universal translator for videos in different languages. The goal is to harness the true potential of AI by personalizing and making it a private AI model. The shift with AI is evident in these new features, which aim to provide a more intuitive and contextual experience for users. With AI becoming an ever-evolving and personalized form of memory, Google is moving boldly but responsibly in integrating AI into various aspects of our lives.', 'topic_source': 'https://www.youtube.com/watch?v=MT8qGJZs4Gw', 'topic_publish_date': '2023-05-11 00:00:00', 'topic_timestamp': '0:00:00'},
        #{'topic_title': 'AI-driven tool for personalizing 3D-printable models', 'topic_summary': 'Researchers have developed an AI-driven tool called Style2Fab that allows users to easily personalize 3D-printable models without compromising their functionality. This tool is particularly useful for creating customized assistive devices. With Style2Fab, makers can quickly customize the design of 3D-printable objects to meet individual needs and preferences. This advancement in AI technology makes it easier for individuals to access personalized solutions and improve their quality of life.', 'topic_source': 'https://news.mit.edu/topic/artificial-intelligence2', 'topic_publish_date': '2023-09-15'},
 #{'topic_title': 'Personalized AI', 'topic_summary': "Google is working on developing personalized AI that can understand users and their surroundings to provide the best results. This AI can help guide decision making, manage workloads, and provide tailored responses in the user's own voice. The more the user interacts with the AI, the more it can assist them in various aspects of their life. The AI can also provide recommendations based on the user's preferences and needs, such as dietary restrictions. As AI continues to advance, it has the potential to transform many aspects of our lives in unimaginable ways.", 'topic_source': 'https://www.youtube.com/watch?v=gMsQO5u7-NQ', 'topic_publish_date': '2023-05-09 00:00:00', 'topic_timestamp': 'The timestamp when the speakers started talking about personalized AI is at 0:01:50.'}
]
    
    ai_agent.set_tweets(tweets=tweets)
    print(ai_agent.get_tweets())

    # 8. Tweet it using scheduler
    
    for tweet in ai_agent.get_tweets():
        tweeterAgent(tweet_details=tweet)


    



if __name__ == "__main__":
   
   main()


    


#uvicorn main:app --host 0.0.0.0 --port 8000




   
