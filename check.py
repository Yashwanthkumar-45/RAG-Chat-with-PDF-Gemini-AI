import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv('AIzaSyAFKkE_PlGcOMGbzWzy12_-QCYgmqw4Fms'))

try:
    available_models = genai.list_models()
    print("Available Models:")
    for model in available_models:
        print(model.name)
except Exception as e:
    print("Error fetching models:", e)
