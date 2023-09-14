from langchain.agents import load_tools, Agent
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI


class AIAgent(Agent):
    def __init__(self):
        super().__init__()
        self.youtube_tool = YouTubeSearchTool()
        self.serp_tool = SerpSearchTool()

    def process(self, input_dict):
        # Use the YouTube search tool to search for videos
        video_urls = self.youtube_tool.search(input_dict['query'], num_videos=10)

        # Use the Serp search tool to search for relevant pages
        serp_results = self.serp_tool.search(input_dict['query'])

        # Combine the video and serp results
        combined_results = video_urls + serp_results

        # Return the combined results
        return combined_results