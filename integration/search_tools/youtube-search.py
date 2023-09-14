#! pip install youtube_search

from langchain.tools import YouTubeSearchTool

# function for getting relevant youtube videos given a user query
# The input of this function is the query and the number of links needed to be fetched
# The output of this function is a list of unstructured youtube links
def get_relevant_youtube_videos(query, num=20): 

    video_urls=[]   

    tool = YouTubeSearchTool()

    search_query = f"{query},{num}" 
    
    result= tool.run(search_query)
    cleaned_result=result[1:-1]

    video_urls = cleaned_result.split(',')
    
    
    return  video_urls
# end def


# function for converting the videos gotten from search tool to valid youtube urls
# The input of the function is a list of unstructured links
# The output of the function is a list of valid youtube urls
def get_youtube_URLS(search_links):

    valid_youtube_URLS=[]
    
    for i,link in enumerate(search_links):
        if(i==0):
            cleaned_link= link[1:-2]
        else:
            cleaned_link= link[2:-1]
        valid_url= f"https://www.youtube.com{cleaned_link}"  
        valid_youtube_URLS.append(valid_url)
    
    return valid_youtube_URLS
#end def





query="What is new about AI?"
num_of_videos=10

unstructured_links= get_relevant_youtube_videos(query, num_of_videos)


youtube_urls=get_youtube_URLS(unstructured_links)

print(youtube_urls)



