
# Supporting libraries
import os
from dotenv import load_dotenv, find_dotenv

#local
from agent.ai_agent import AIAgent

if __name__ == "__main__":

    load_dotenv(find_dotenv())
    ai_agent = AIAgent()

    # 1. Search

    #ai_agent.search_ai_news_urls()   #comment for now to not use up much requests

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
    print(ai_agent.get_tweets())

    # 8. Tweet it using scheduler






    #trial
    # video_info= ai_agent.get_youtube_info(youtube_link="https://www.youtube.com/watch?v=LK5j3pp0Too&pp=ygUbV2hhdCBpcyBuZXcgYWJvdXQgQUkgdG9kYXk'")
    # publish_date= video_info["publish_date"]
    # author= video_info["author"]
    # print(publish_date)
    # print(author)

    # website_publish= ai_agent.get_website_publish_date(website_link="https://news.mit.edu/topic/artificial-intelligence2")
    # print(website_publish)
