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
from flask import Flask, render_template, request, jsonify
# from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.schema.agent import AgentFinish
from langchain.schema.runnable import RunnablePassthrough

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

class OpenGoogleMapInput:
    def __init__(self, latitude: float, longitude: float):
        self.latitude = latitude
        self.longitude = longitude

@tool
def get_Restaurant_Info(latitude: float, longitude: float) -> dict:
    """Fetch current restaurant for given coordinates."""

    BASE_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

    params = {
        'location': f"{latitude},{longitude}",
        'radius': 1000,
        'type': ['restaurant'],
        'key': api_key,
        'language': 'zh-TW'
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
            
        return f'附近有一些不錯的餐廳可以去試試看，包括{restaurants[0]["name"]}({restaurants[0]["rating"]}星)、{restaurants[1]["name"]}({restaurants[1]["rating"]}星)、{restaurants[2]["name"]}({restaurants[2]["rating"]}星)、{restaurants[3]["name"]}({restaurants[3]["rating"]}星)、{restaurants[4]["name"]}({restaurants[4]["rating"]}星)等。'
    else:
        raise Exception(f"API Request failed with status code: {response.status_code}")


@tool
def get_Cafe_Info(latitude: float, longitude: float) -> dict:
    """Fetch current restaurant for given coordinates."""

    BASE_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

    params = {
        'location': f"{latitude},{longitude}",
        'radius': 2000,
        'type': ['cafe'],
        'key': api_key,
        'language': 'zh-TW'
    }
    # Make the request
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code == 200:
        results = response.json()
        cafe = []
        count = 0
        for result in results.get('results', []):
            if count >= 5:
                break
            cafe_info = {
                'name': result.get('name', ''),
                'location': result.get('vicinity', ''),
                'rating': result.get('rating', ''),
                # 可根据需要提取其他信息
            }
            cafe.append(cafe_info)
            count = count + 1
              
        return f'附近有一些不錯的咖啡廳可以去試試看，包括{cafe[0]["name"]}({cafe[0]["rating"]}星)、{cafe[1]["name"]}({cafe[1]["rating"]}星)、{cafe[2]["name"]}({cafe[2]["rating"]}星)、{cafe[3]["name"]}({cafe[3]["rating"]}星)、{cafe[4]["name"]}({cafe[4]["rating"]}星)等。'
    else:
        raise Exception(f"API Request failed with status code: {response.status_code}")


@tool
def get_tourist_attraction_Info(latitude: float, longitude: float) -> dict:
    """Fetch current restaurant for given coordinates."""

    BASE_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

    params = {
        'location': f"{latitude},{longitude}",
        'radius': 8000,
        'type': ['tourist_attraction'],
        'key': api_key,
        'language': 'zh-TW'
    }
    # Make the request
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code == 200:
        results = response.json()
        tourist_attraction = []
        count = 0
        for result in results.get('results', []):
            if count >= 5:
                break
            tourist_attraction_info = {
                'name': result.get('name', ''),
                'location': result.get('vicinity', ''),
                'rating': result.get('rating', ''),
                # 可根据需要提取其他信息
            }
            tourist_attraction.append(tourist_attraction_info)
            count = count + 1
            
        return f'附近有一些不錯的旅遊景點可以去試試看，包括{tourist_attraction[0]["name"]}({tourist_attraction[0]["rating"]}星)、{tourist_attraction[1]["name"]}({tourist_attraction[1]["rating"]}星)、{tourist_attraction[2]["name"]}({tourist_attraction[2]["rating"]}星)、{tourist_attraction[3]["name"]}({tourist_attraction[3]["rating"]}星)、{tourist_attraction[4]["name"]}({tourist_attraction[4]["rating"]}星)等。'
    else:
        raise Exception(f"API Request failed with status code: {response.status_code}")

tools = [get_Cafe_Info, get_Restaurant_Info, get_tourist_attraction_Info]

functions = [format_tool_to_openai_function(f) for f in tools]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful assistant who can recommend place to go"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])


model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0).bind(functions=functions)

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
            # return result
            return intermediate_steps[-1][1]
        tool = {
            "get_Restaurant_Info": get_Restaurant_Info, 
            "get_Cafe_Info": get_Cafe_Info,
            "get_tourist_attraction_Info": get_tourist_attraction_Info
        }[result.tool]
        
        observation = tool.run(result.tool_input)
        intermediate_steps.append((result, observation))
# chatbot_response = run_agent("台南成功大學附近美食")
# print(chatbot_response)

@app.route('/LLMChat1', methods=['POST'])
def LLMChat():
    user_input = request.json.get('message')
    if user_input:
        response = run_agent(user_input)
        return jsonify({'response': response})
    else:
        return jsonify({'response': 'Error: No message provided'}), 400

if __name__ == '__main__':
    app.run(debug=True)