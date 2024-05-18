import os
import openai
from dotenv import load_dotenv, find_dotenv
from dotenv import dotenv_values
env_vars = dotenv_values()
api_key = env_vars.get('GOOGLE_MAPS_API_KEY')
GOOGLE_CX = env_vars.get('GOOGLE_CX')
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
from langchain.memory import ConversationBufferMemory


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/chatbot1')
def chatbot1():
    return render_template('index1.html')

@app.route('/chatbot2')
def chatbot2():
    return render_template('index.html')

@app.route('/chatbot3')
def chatbot3():
    return render_template('index2.html')

class OpenGoogleMapInput:
    def __init__(self, latitude: float, longitude: float):
        self.latitude = latitude
        self.longitude = longitude

@tool
def get_nearby_places(latitude: float, longitude: float, keyword: str) -> dict:
    """Fetch nearby places based on given coordinates and keyword."""
    
    BASE_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

    params = {
        'location': f"{latitude},{longitude}",
        'radius': 500,
        'keyword': keyword,
        'key': api_key,
        'language': 'zh-TW'
    }
    # Make the request
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code == 200:
        results = response.json()
        places_info = []
        for result in results.get('results', [])[:10]:  # 只處理前五個結果
            place_info = {
                'name': result.get('name', ''),
                'location': result.get('vicinity', ''),
                'rating': result.get('rating', ''),
                # 可根據需要提取其他信息
            }
            places_info.append(place_info)
        
        if places_info:
            places_info_str = ", ".join([f"{place['name']}({place['rating']}星)" for place in places_info])
            return f'附近有一些不錯的{keyword}，包括{places_info_str}等。'
        else:
            return f'附近沒有找到{keyword}。'
    else:
        raise Exception(f"API request failed with status code: {response.status_code}")

# 取得某地點附近的資訊與各自評論
@tool
def get_reviews_nearby(latitude: float, longitude: float, keyword: str) -> str:
    """Fetch reviews for places near given coordinates based on keyword."""
    output = ""
    
    BASE_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

    params = {
        'location': f"{latitude},{longitude}",
        'radius': 500,
        'keyword': keyword,
        'key': api_key,
        'language': 'zh-TW'
    }
    # Make the request
    response = requests.get(BASE_URL, params=params)
    if response.status_code != 200:
        raise Exception(f"API Request failed with status code: {response.status_code}")

    results = response.json().get('results', [])[:5]
    output = ""

    for result in results:
        place_name = result.get('name', '')
        place_id = result.get('place_id', '')

        if not place_id:
            output += f"找不到名為 {place_name} 的地點。\n\n"
            continue
    
        reviews_url = f"https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&fields=name,reviews&key={api_key}"
        reviews_response = requests.get(reviews_url)
    
        if reviews_response.status_code != 200:
            raise Exception(f"Reviews API Request failed with status code: {reviews_response.status_code}")
        
        reviews_data = reviews_response.json()
        place_reviews_info = reviews_data.get('result', {}).get('reviews', [])
        
        if place_reviews_info:
            output += f"{place_name}\n"
            for review_info in place_reviews_info[:5]:
                output += f"評論者: {review_info.get('author_name', '')}, 評分: {review_info.get('rating', '')}星, 評論: {review_info.get('text', '')}\n"
            output += "\n"
        else:
            output += f"{place_name} 目前還沒有評論。\n\n"

    return output

   

# 給某一特定定點的評價
@tool
def get_place_reviews(place_name: str) -> str:
    """Fetch reviews for a specific place."""
    
    BASE_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"

    params = {
        'query': place_name,
        'key': api_key,
        'language': 'zh-TW'
    }
    # Make the request
    response = requests.get(BASE_URL, params=params)

    if response.status_code != 200:
        raise Exception(f"Text Search API Request failed with status code: {response.status_code}")

    results = response.json().get('results', [])
    if not results:
        return f"找不到名為 {place_name} 的地點。"

    place_id = results[0].get('place_id', '')
    if not place_id:
        return f"找不到名為 {place_name} 的地點。"

    reviews_url = f"https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&fields=name,reviews&key={api_key}"
    reviews_response = requests.get(reviews_url)
    if reviews_response.status_code != 200:
        raise Exception(f"Reviews API Request failed with status code: {reviews_response.status_code}")

    reviews_data = reviews_response.json()
    place_reviews_info = reviews_data.get('result', {}).get('reviews', [])
    if not place_reviews_info:
        return f"{place_name} 目前還沒有評論。"

    output = f"{place_name} 的評論如下：\n\n"
    for review_info in place_reviews_info:
        output += f"評論者: {review_info.get('author_name', '')}, 評分: {review_info.get('rating', '')}星, 評論: {review_info.get('text', '')}\n"
    return output


# 僅能處理關鍵字查詢無法回答更針對性的問題
def google_search(query):
    url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={GOOGLE_CX}&q={query}&num=5&language=zh-TW"
    response = requests.get(url)
    if response.status_code == 200:
        search_results = response.json().get('items', [])
        return [result['snippet'] for result in search_results]
    else:
        raise Exception(f"Google Custom Search API Request failed with status code: {response.status_code}")

@tool
def get_google_search(keyword: str) -> str:
    """Fetch information based on keyword."""
    output = ""
    
    search_results = google_search(keyword)
    if search_results:
        for result in search_results:
            output += result + "\n"
        return output
    else:
        return "未找到相關資訊。"


tools = [get_nearby_places, get_place_reviews, get_reviews_nearby, get_google_search]
functions = [format_tool_to_openai_function(f) for f in tools]
prompt = ChatPromptTemplate.from_messages([
    (   
        "system", 
        "You are a versatile assistant with variable response methods, who can search google information for any question and u can recommend user any place to go from google map"
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

prompt1 = ChatPromptTemplate.from_messages([
    (   
        "system", 
        "You are a sweet and cuddly assistant always give cute response, who can search google information for any question and u can recommend user any place to go from google map"
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

prompt2 = ChatPromptTemplate.from_messages([
    (   
        "system", 
        "You are a Rude and Impolite and Aggressive assistant with variable response methods, who can search google information for any question and u can recommend user any place to go from google map"
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

llm = ChatOpenAI(model="gpt-4o-2024-05-13", temperature=1.0)

agent = create_tool_calling_agent(llm, tools, prompt)
agent1 = create_tool_calling_agent(llm, tools, prompt1)
agent2 = create_tool_calling_agent(llm, tools, prompt2)

memory = ConversationBufferMemory(return_messages=True,memory_key="chat_history")

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory)  
agent_executor1 = AgentExecutor(agent=agent1, tools=tools, verbose=True, memory=memory)
agent_executor2 = AgentExecutor(agent=agent2, tools=tools, verbose=True, memory=memory)
def run_agent(user_input):
    result = agent_executor.invoke({
            "input": user_input, 
    })
    return result 

def run_agent1(user_input):
    result = agent_executor1.invoke({
            "input": user_input, 
    })
    return result

def run_agent2(user_input):
    result = agent_executor2.invoke({
            "input": user_input, 
    })
    return result

def format_response(response):
    formatted_response = response['output']
    
    # 將換行符號替換為HTML的換行標籤
    formatted_response = formatted_response.replace('\n\n\n\n', '<br>')  # 替換多個換行為一個HTML換行標籤
    formatted_response = formatted_response.replace('\n', '<br>')  # 替換單個換行為一個HTML換行標籤
    
    # 將標題添加到各區域
    formatted_response = formatted_response.replace('###', '<h3>')  # 將###替換為HTML標題標籤
    formatted_response = formatted_response.replace('###', '</h3>')  # 將###替換為HTML標題標籤

    formatted_response = formatted_response.replace('**', '<h3>')
    formatted_response = formatted_response.replace('**', '</h3>')
    
    # 將回應文本的字體大小設置為較小的尺寸
    formatted_response = '<span style="font-size: smaller;">' + formatted_response + '</span>'
    
    return formatted_response

@app.route('/LLMChat', methods=['POST'])
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

@app.route('/LLMChat1', methods=['POST'])
def LLMChat2():
    user_input = request.json.get('message')
    if user_input:
        response = run_agent1(user_input)
        formatted_response = format_response(response)
        # 去掉回應中的引號
        formatted_response = formatted_response.replace('"', '')
        return jsonify({'response': formatted_response})
    else:
        return jsonify({'response': 'Error: No message provided'}), 400

@app.route('/LLMChat2', methods=['POST'])
def LLMChat3():
    user_input = request.json.get('message')
    if user_input:
        response = run_agent2(user_input)
        formatted_response = format_response(response)
        # 去掉回應中的引號
        formatted_response = formatted_response.replace('"', '')
        return jsonify({'response': formatted_response})
    else:
        return jsonify({'response': 'Error: No message provided'}), 400

if __name__ == '__main__':
    app.run(debug=True)