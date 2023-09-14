from langchain.agents import load_tools, Agent
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

from search_tools import YouTubeVideoFinder, SerpSearchTool

class AIAgent(Agent):
    def __init__(self):
        super().__init__()
        self.youtube_tool = YouTubeVideoFinder("What is new about AI today?", 10)
        self.serp_tool = SerpSearchTool("What is new about AI today?")
        self.youtube_urls=[]
        self.website_urls=[]

    def search_ai_urls(self, input_dict):
        # Use the YouTube search tool to search for videos
        video_urls = self.youtube_tool.find_videos()
        self.youtube_urls= self.youtube_tool.get_youtube_urls()

        # Use the Serp search tool to search for relevant pages
        websites_url_data = self.serp_tool.run(input_dict['query'])  #json

        for i,website in enumerate(websites_url_data):
            self.website_urls.append(website[i]['link'])
        #print(response_data['news'][0]['link'])


        # # Return the combined results
        # return youtube_urls, websites_url

    def create_faiss_vectorstore(self):
        pass


    def extract_websites_topics(self):
        pass

    def extract_youtube_topics(self):
        pass

    def summarize_website_topics(self):
        pass

    def summarize_youtube_topics(self):
        pass

    def get_tweet_info(self):
        pass