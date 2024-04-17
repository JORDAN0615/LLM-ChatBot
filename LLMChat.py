from flask import Flask, render_template, request, jsonify
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from langchain.prompts.prompt import PromptTemplate

app = Flask(__name__)

# 加载 .env 文件中的环境变量
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
print(os.environ["OPENAI_API_KEY"])
# 创建 ChatOpenAI 模型实例
model = ChatOpenAI(model="gpt-3.5-turbo")

# Flask 路由定义
@app.route('/')
def home():
    return render_template('index.html')

def chat_with_bot(sentence):
    # 创建 ChatPromptTemplate 实例
    prompt_template = """
    Answer language must be traditional Chinese 
    MESSAGE: {sentence}
    ANSWER:
    """
    prompt = PromptTemplate.from_template(prompt_template)
    
    # 创建 StrOutputParser 实例
    output_parser = StrOutputParser()

    # 创建 langchain，将 prompt、model 和 output parser 链接起来
    chain = prompt | model | output_parser
    
    result = chain.invoke({"sentence": sentence})
    return result


@app.route('/LLMChat', methods=['POST'])
def LLMChat():
    user_input = request.json.get('message')
    if user_input:
        response = chat_with_bot(user_input)
        return jsonify({'response': response})
    else:
        return jsonify({'response': 'Error: No message provided'}), 400

if __name__ == '__main__':
    app.run(debug=True)