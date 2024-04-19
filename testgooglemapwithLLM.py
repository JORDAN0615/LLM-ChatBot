import os
import openai
from dotenv import load_dotenv, find_dotenv
from dotenv import dotenv_values
env_vars = dotenv_values()
api_key = env_vars.get('GOOGLE_MAPS_API_KEY')

_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

from langchain.tools import tool
import requests
# from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.schema.agent import AgentFinish
from langchain.schema.runnable import RunnablePassthrough
class OpenGoogleMapInput:
    def __init__(self, latitude: float, longitude: float):
        self.latitude = latitude
        self.longitude = longitude

@tool
def get_Info(latitude: float, longitude: float) -> dict:
    """Fetch current restaurant for given coordinates."""

    BASE_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

    params = {
        'location': f"{latitude},{longitude}",
        'radius': 1000,
        'type': 'restaurant',
        'key': api_key,
    }
    # Make the request
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code == 200:
        results = response.json()
        restaurants = []
        count = 0
        for result in results.get('results', []):
            if count >= 5:
                break
            restaurant_info = {
                'name': result.get('name', ''),
                'location': result.get('vicinity', ''),
                'rating': result.get('rating', ''),
                # 可根据需要提取其他信息
            }
            restaurants.append(restaurant_info)
            count = count + 1
        return f'There is some nice {restaurants}°'
    else:
        raise Exception(f"API Request failed with status code: {response.status_code}")


tools = [get_Info]

functions = [format_tool_to_openai_function(f) for f in tools]
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0).bind(functions=functions)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful but sassy assistant"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

chain = prompt | model | OpenAIFunctionsAgentOutputParser()

agent_chain = RunnablePassthrough.assign(
    agent_scratchpad= lambda x: format_to_openai_functions(x["intermediate_steps"])
) | chain

def run_agent(user_input):
    intermediate_steps = []
    while True:
        result = agent_chain.invoke({
            "input": user_input, 
            "intermediate_steps": intermediate_steps
        })
        if isinstance(result, AgentFinish):
            return result
        tool = {
            "get_Info": get_Info,   
        }[result.tool]
        
        observation = tool.run(result.tool_input)
        intermediate_steps.append((result, observation))
print(run_agent("西門站附近美食"))