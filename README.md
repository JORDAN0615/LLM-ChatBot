# LangChain Chatbot with Google Maps Integration

## Overview
This project implements a chatbot using OpenAI's GPT-3.5 language model integrated with Google Maps API to provide information about nearby restaurants, cafes, bars, and tourist attractions. The chatbot is built using Python and Flask.

## Features
- Provides recommendations for restaurants, cafes, bars, and tourist attractions based on user input.
- Supports both English and Chinese input and output.
- Utilizes Google Maps API to fetch real-time information about nearby places.
- Interactive interface for users to input queries and receive responses from the chatbot.

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/your_username/langchain-chatbot.git
   ```
2. Navigate to the project directory:
   ```
   cd langchain-chatbot
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Set up environment variables:
   - Create a `.env` file in the root directory.
   - Add your Google Maps API key and OpenAI API key to the `.env` file:
     ```
     GOOGLE_MAPS_API_KEY=your_google_maps_api_key
     OPENAI_API_KEY=your_openai_api_key
     ```

## Usage
1. Run the Flask app:
   ```
   python testgooglemapwithLLM.py
   ```
2. Access the chatbot interface in your web browser at `http://localhost:5000/`.
3. Enter your query in the chatbox and press Enter.
4. Receive recommendations and information from the chatbot.

## Credits
- This project utilizes the LangChain library for building chatbots and integrating with OpenAI.
- Developed by [Jordan Tseng](https://github.com/JORDAN0615).

## License
This project is licensed under the [MIT License](LICENSE).

---

Feel free to customize the README to include more details about your project or provide additional instructions as needed.
