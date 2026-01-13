from google import genai
from langsmith import traceable
import os
from dotenv import load_dotenv
load_dotenv()

# Inicializar cliente de Gemini
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

@traceable # Auto-trace this function
def pipeline(user_input: str):
    result = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_input
    )
    return result.candidates[0].content.parts[0].text

print(pipeline("¡Hola! ¿Cómo estás?"))
