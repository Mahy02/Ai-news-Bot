from langchain.tools import YouTubeSearchTool

class YouTubeVideoFinder:
    def __init__(self, query, num_of_videos):
        self.query = query
        self.num_of_videos = num_of_videos
        self.video_urls = []

    def find_videos(self):
        tool = YouTubeSearchTool()
        search_query = f"{self.query},{self.num_of_videos}"
        result = tool.run(search_query)
        cleaned_result = result[1:-1]
        self.video_urls = cleaned_result.split(',')
        return self.video_urls

    def get_youtube_urls(self):
        valid_youtube_URLS = []
        for i, link in enumerate(self.video_urls):
            if i == 0:
                cleaned_link = link[1:-2]
            else:
                cleaned_link = link[2:-1]
            valid_url = f"https://www.youtube.com{cleaned_link}"
            valid_youtube_URLS.append(valid_url)
        return valid_youtube_URLS






# query = "What is new about AI?"
# num_of_videos = 10

# video_finder = YouTubeVideoFinder(query, num_of_videos)
# video_urls = video_finder.find_videos()
# youtube_urls = video_finder.get_youtube_urls()
