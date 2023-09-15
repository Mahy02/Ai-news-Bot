# from langchain
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage

# from pydantic
from pydantic import BaseModel

# from fastapi
from fastapi import FastAPI

# python
import os
import requests
import json
from dotenv import load_dotenv


# local
from twit import tweeter

load_dotenv()
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
Use links sparingly and only when really needed, but when you do make sure you actually include them AND ONLY PUT THE LINk, dont put brackets around them. 
Only return the thread, no other text, and make each tweet its own paragraph.
Make sure each tweet is lower that 220 chars
    Topic Headline:{topic}
    Info: {info}
    """

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

#twitapi is the client with all keys and secrets
twitapi = tweeter()

def tweetertweet(thread):

    tweets = thread.split("\n\n")
   
    #check each tweet is under 280 chars
    for i in range(len(tweets)):
        if len(tweets[i]) > 280:    
            prompt = f"Shorten this tweet to be under 280 characters: {tweets[i]}"
            tweets[i] = llm3.predict(prompt)[:280]
    #give some spacing between sentances
    tweets = [s.replace('. ', '.\n\n') for s in tweets]

    for tweet in tweets:
        tweet = tweet.replace('**', '')

    try:
        response = twitapi.create_tweet(text=tweets[0])
        id = response.data['id']
        tweets.pop(0)
        for i in tweets:
            print("tweeting: " + i)
            reptweet = twitapi.create_tweet(text=i, 
                                    in_reply_to_tweet_id=id, 
                                    )
            id = reptweet.data['id']
        return "Tweets posted successfully"
    except Exception as e:
        return f"Error posting tweets: {e}"





# 5. Set this as an API endpoint via FastAPI
app = FastAPI()





@app.post("/")
def researchAgent(tweet_details):
    topic_summary= tweet_details['topic_summary']
    topic_headline= tweet_details['topic_title']
    thread = llm_chain.predict(info = topic_summary, topic = topic_headline)
    #we should also add source, publish date and timstamp here at the end of the thread and pass it to tweetertweet()
    ret = tweetertweet(thread)
    return ret