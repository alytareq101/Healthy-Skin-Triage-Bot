# test_llm.py
from dotenv import load_dotenv
import os
from openai import OpenAI

# Load the .env file
load_dotenv()

# Get your API key
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Example prompt
prompt_text = "Explain why high-risk skin lesions should be referred to a dermatologist."


# Call the model
response = client.responses.create(
    model="gpt-5-nano",
    input=prompt_text,
    store=True  # optional, you can set False if you don't want to store the response
)

# Print the generated text
print("LLM Output:\n", response.output_text)
