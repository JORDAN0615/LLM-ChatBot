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
from langchain.agents import AgentExecutor, create_tool_calling_agent


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

class OpenGoogleMapInput:
    def __init__(self, latitude: float, longitude: float):
        self.latitude = latitude
        self.longitude = longitude

@tool
def get_restaurant_info(latitude: float, longitude: float) -> dict:
    """Fetch current restaurants for given coordinates."""

    BASE_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

    params = {
        'location': f"{latitude},{longitude}",
        'radius': 1000,
        'keyword': '餐廳',  # 使用關鍵字搜索
        'key': api_key,
        'language': 'zh-TW'
    }
    # Make the request
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code == 200:
        results = response.json()
        restaurants = []
        for result in results.get('results', [])[:5]:  # 只處理前五個結果
            restaurant_info = {
                'name': result.get('name', ''),
                'location': result.get('vicinity', ''),
                'rating': result.get('rating', ''),
                # 可根据需要提取其他信息
            }
            restaurants.append(restaurant_info)
        
        if restaurants:
            restaurant_info_str = ", ".join([f"{restaurant['name']}({restaurant['rating']}星)" for restaurant in restaurants])
            return f'附近有一些不錯的餐廳可以去試試看，包括{restaurant_info_str}等。'
        else:
            return '附近沒有找到餐廳。'
    else:
        raise Exception(f"API Request failed with status code: {response.status_code}")



@tool
def get_ramen(latitude: float, longitude: float) -> dict:
    """Fetch current ramen restaurants for given coordinates."""

    BASE_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

    params = {
        'location': f"{latitude},{longitude}",
        'radius': 800,
        'keyword': '拉麵',  # 修改為單個關鍵字字符串
        'key': api_key,
        'language': 'zh-TW'
    }
    # Make the request
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code == 200:
        results = response.json()
        ramen_info = []
        for result in results.get('results', [])[:5]:  # 只處理前五個結果
            restaurant_info = {
                'name': result.get('name', ''),
                'location': result.get('vicinity', ''),
                'rating': result.get('rating', ''),
                # 可根據需要提取其他信息
            }
            ramen_info.append(restaurant_info)
        
        if ramen_info:
            restaurant_info_str = ", ".join([f"{restaurant['name']}({restaurant['rating']}星)" for restaurant in ramen_info])
            return f'附近有一些不錯的拉麵店可以去試試看，包括{restaurant_info_str}等。'
        else:
            return '附近沒有找到拉麵店。'
    else:
        raise Exception(f"API Request failed with status code: {response.status_code}")



@tool
def get_porkRice(latitude: float, longitude: float) -> dict:
    """Fetch current restaurant for given coordinates."""

    BASE_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

    params = {
        'location': f"{latitude},{longitude}",
        'radius': 800,
        'keyword': '滷肉飯',  # 修改為單個關鍵字字符串
        'key': api_key,
        'language': 'zh-TW'
    }
    # Make the request
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code == 200:
        results = response.json()
        porkRice_info = []
        for result in results.get('results', [])[:5]:  # 只處理前五個結果
            restaurant_info = {
                'name': result.get('name', ''),
                'location': result.get('vicinity', ''),
                'rating': result.get('rating', ''),
                # 可根據需要提取其他信息
            }
            porkRice_info.append(restaurant_info)
        
        if porkRice_info:
            restaurant_info_str = ", ".join([f"{info['name']}({info['rating']}星)" for info in porkRice_info])
            return f'附近有一些不錯的滷肉飯可以去試試看，包括{restaurant_info_str}等。'
        else:
            return '附近沒有找到滷肉飯店。'
    else:
        raise Exception(f"API Request failed with status code: {response.status_code}")

@tool
def get_Cafe_Info(latitude: float, longitude: float) -> dict:
    """Fetch current cafe for given coordinates."""

    BASE_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

    params = {
        'location': f"{latitude},{longitude}",
        'radius': 800,
        'keyword': '咖啡廳',  # 修改為單個關鍵字字符串
        'key': api_key,
        'language': 'zh-TW'
    }
    # Make the request
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code == 200:
        results = response.json()
        cafe_info = []
        for result in results.get('results', [])[:5]:  # 只處理前五個結果
            cafe = {
                'name': result.get('name', ''),
                'location': result.get('vicinity', ''),
                'rating': result.get('rating', ''),
                # 可根據需要提取其他信息
            }
            cafe_info.append(cafe)
        
        if cafe_info:
            cafe_info_str = ", ".join([f"{cafe['name']}({cafe['rating']}星)" for cafe in cafe_info])
            return f'附近有一些不錯的咖啡廳可以去試試看，包括{cafe_info_str}等。'
        else:
            return '附近沒有找到咖啡廳。'
    else:
        raise Exception(f"API Request failed with status code: {response.status_code}")


@tool
def get_tourist_attraction_Info(latitude: float, longitude: float) -> dict:
    """Fetch current tourist attractions for given coordinates."""

    BASE_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

    params = {
        'location': f"{latitude},{longitude}",
        'radius': 5000,
        'keyword': '旅遊景點',  # 修改為單個關鍵字字符串
        'key': api_key,
        'language': 'zh-TW'
    }
    # Make the request
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code == 200:
        results = response.json()
        tourist_attraction_info = []
        for result in results.get('results', [])[:5]:  # 只處理前五個結果
            attraction = {
                'name': result.get('name', ''),
                'location': result.get('vicinity', ''),
                'rating': result.get('rating', ''),
                # 可根據需要提取其他信息
            }
            tourist_attraction_info.append(attraction)
        
        if tourist_attraction_info:
            attraction_info_str = ", ".join([f"{attraction['name']}({attraction['rating']}星)" for attraction in tourist_attraction_info])
            return f'附近有一些不錯的旅遊景點可以去試試看，包括{attraction_info_str}等。'
        else:
            return '附近沒有找到旅遊景點。'
    else:
        raise Exception(f"API Request failed with status code: {response.status_code}")


tools = [get_Cafe_Info, get_restaurant_info, get_tourist_attraction_Info, get_ramen, get_porkRice]
functions = [format_tool_to_openai_function(f) for f in tools]
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful assistant who can recommend place to go"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def run_agent(user_input):
    result = agent_executor.invoke({
            "input": user_input, 
    })
    return result  

def format_response(response):
    # 將換行符號插入到回應文本中
    formatted_response = response['output'].replace('\n', '\n\n')  # 在每個換行前插入一個額外的換行符號
    return formatted_response

# input_message = "推薦花蓮一日行程"
# chatbot_response = run_agent(input_message)
# formatted_response = format_response(chatbot_response)
# print(formatted_response)

@app.route('/LLMChat1', methods=['POST'])
def LLMChat():
    user_input = request.json.get('message')
    if user_input:
        response = run_agent(user_input)
        formatted_response = format_response(response)
        # 去掉回應中的引號
        formatted_response = formatted_response.replace('"', '')
        return jsonify({'response': formatted_response})
    else:
        return jsonify({'response': 'Error: No message provided'}), 400

if __name__ == '__main__':
    app.run(debug=True)