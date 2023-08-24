from langchain.chat_models import ChatOpenAI
#from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.llms import OpenAI
from langchain import SerpAPIWrapper
from langchain.agents.tools import Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, HuggingFaceHub
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.agents import load_tools, initialize_agent
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline
import torch
#import openai
import os

from dotenv import load_dotenv


# Load environment variables from .env
load_dotenv()

# Access environment variables
serp_api_key = os.environ.get('SERPAPI_API_KEY')
huggingface_token = os.environ.get('HUGGINGFACEHUB_API_TOKEN')


model = "tiiuae/falcon-7b-instruct" #tiiuae/falcon-40b-instruct

tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = pipeline(
    "text-generation", #task
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})

template = """
You are an intelligent chatbot. Help the following question with brilliant answers.
Question: {question}
Answer:"""
prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "Explain what is Artificial Intellience as Nursery Rhymes "

print(llm_chain.run(question))





# llm = OpenAI(temperature=0)

# tools = load_tools(["serpapi", "llm-math"], llm=llm)

# agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# agent.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?")