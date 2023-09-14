from langchain.tools import Tool
from langchain.tools import BraveSearch
from langchain.tools import DuckDuckGoSearchRun
from langchain.tools import DuckDuckGoSearchResults
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.utilities import SerpAPIWrapper

import os
import pprint
import requests
import json

from dotenv import load_dotenv


# Load environment variables from .env
load_dotenv()

# Access environment variables
brave_api_key= os.environ.get('BRAVE_API_KEY')
serp_api_key = os.environ.get('SERPAPI_API_KEY')
huggingface_token = os.environ.get('HUGGINGFACEHUB_API_TOKEN')
google_api_key=os.environ.get("GOOGLE_API_KEY")
search_engine_id=os.environ.get("GOOGLE_CSE_ID")

os.environ["SERPAPI_API_KEY"]= os.environ.get('SERPAPI_API_KEY')
os.environ["SERPER_API_KEY"]=os.environ.get('SERPER_API_KEY')
os.environ["HUGGINGFACEHUB_API_TOKEN"]= os.environ.get('HUGGINGFACEHUB_API_TOKEN')
os.environ["GOOGLE_CSE_ID"] = os.environ.get("GOOGLE_CSE_ID")
os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY")


# Testing different search tools:
###################################

##############################################################################################################
#1. DuckDuck go


# # a. using DuckDuckGoSearch

def search_with_duckduckgo_search(query):
    search = DuckDuckGoSearchRun()
    print(f"first: {search.run(query)}")

def search_with_duckduckgo_search_results(query):
    search = DuckDuckGoSearchResults(backend="news")
    print(f"second: {search.run(query)}")


# #c. using Wrapper for more control  -- not working

# print("----------------------------")
# wrapper = DuckDuckGoSearchAPIWrapper(max_results=2)
# print(wrapper)
# search = DuckDuckGoSearchResults(api_wrapper=wrapper, backend="news")
# print(search.json)
# print("third:       ")
# print(search.run("What is new about AI?"))


##############################################################################################################

# Serp API with google engine  - BEST ONE IN ALL

#BEHIND THE HOOD where we use googleserp api   ---I think it does exactly the same thing
def searchSerp(query):
    url = "https://google.serper.dev/search"
    payload=json.dumps({
        'q': query,
        'type': "news",
        'num': 10,
        'hl': 'en',
        #"key": google_api_key,
        #"cx":search_engine_id,
        #"dateRestrict": "2023-08-01:2023-08-28"
    })
    headers= {
        'X-API-KEY': serp_api_key,
        'Content-Type': 'application/json'
    }
    response= requests.request("POST", url, headers=headers, data=payload)
    response_data= response.json()
    #pprint.pp("Search results: ", response_data)
    print("Search pars: ", response_data['searchParameters'])
    #news if type is news, and organic if type is search
    for i,item in enumerate(response_data['news']):
        print(response_data['news'][i])
    return response_data


#what it looks like:
# Search pars:  {'q': 'What is new about AI today?', 'hl': 'en', 'num': 3, 'type': 'news', 'engine': 'google'}
# {'title': "New 'Magisterium AI' adds ecclesial twist to artificial intelligence", 'link': 'https://cruxnow.com/interviews/2023/08/new-magisterium-ai-adds-ecclesial-twist-to-artificial-intelligence/', 'snippet': 'As an AI-based app currently in the beta phase, Magisterium AI “could be a game changer for the Church,” Sanders said.', 'date': '8 hours ago', 'source': 'Crux Now', 'imageUrl': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcST1jeyjIoFLqtETcatnszmOB-64x27KD-4pU2tAN0E-xH75nyD1AAAGEgrKQ&s', 'position': 1}
# {'title': 'AI could choke on its own exhaust as it fills the web', 'link': 'https://www.axios.com/2023/08/28/ai-content-flood-model-collapse', 'snippet': 'AI turbocharges the ability to create mountains of new content ... fear that AI could undermine the jobs of people who create content today,...', 'date': '6 hours ago', 'source': 'Axios', 'imageUrl': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSKUP62l1aTSLm_nYEBmQj0UqFVysVSX9xFl6G4QexHGSJu0NbBpeXUl_730w&s', 'position': 2}
# {'title': 'The Coming Wave by Mustafa Suleyman review – AI, synthetic biology and a new dawn for humanity', 'link': 'https://www.theguardian.com/books/2023/aug/28/the-coming-wave-by-mustafa-suleyman-review-ai-synthetic-biology-and-a-new-dawn-for-humanity', 'snippet': 'Together, he thinks, these two “will usher in a new dawn for humanity ... and now they are working on creating new forms of biological life.', 'date': '7 hours ago', 'source': 'The Guardian', 'imageUrl': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRL6TL-2uNDcP4QX_EzblSLwrOY9Mi4g6NMCtlTulkwFLyLYyPupz9PN795Vg&s', 'position': 3}

#if u want to specify engine use normal serp api:
# params = {
#     "engine": "bing",
#     "gl": "us",
#     "hl": "en",
# }
#search = SerpAPIWrapper(params=params)

##############################################################################################################

# Google Search

def top5_results(query):
    search = GoogleSearchAPIWrapper()
    results = search.results(query, 5)
    return results

def google_search_tool(query):
    # search = GoogleSearchAPIWrapper()
    #results = search.results(query, 5)
    tool = Tool(name="Google Search", description="Search Google for recent results in the AI field.", func=top5_results)
    for i, result in enumerate(tool.run(query)):
        print(tool.run(query)[i])
    return tool.run(query)[i]

##############################################################################################################

# Google Serper API   -best one yet


def google_serper_api(query, k=2, type="news", hl="en"):
    search = GoogleSerperAPIWrapper(k=k, type=type, hl=hl)
    results = search.results(query)
    pprint.pp(results)
    return results


##############################################################################################################

# Brave Search

def braveSearch(inputText):
  brave_api_key = ""
  tool = BraveSearch.from_api_key(api_key=brave_api_key, search_kwargs={"count": 1})
  return (tool.run(inputText))


# IT NEEDS CREDIT CARD INFO XX


##############################################################################################################

# tetsing the search:

# print("Duck Duck Go: \n")
# search_with_duckduckgo_search("What is new about AI?")
# print("\n")

# print("Duck Duck Go  with results: \n")
# search_with_duckduckgo_search_results("What is new about AI?")
# print("\n")


# # sec best ------------------

# print("Google search: \n")
# google_search_tool("what is new about AI today?")
# print("\n")


# #------------------ Best 2:

# print("Google Serper: \n")
# google_serper_api("AI", k=2, type="news", hl="en")
# print("\n")

print("Serper with google engine: \n")
searchSerp("What is new about AI today?")






#Problems need to be resolved:
# 1. Date  "extracting today's date or the date the user asks it about"
# 2. secure websites???





