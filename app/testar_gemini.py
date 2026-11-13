
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client()

try:
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents="Responda apenas: Gemini funcionando!"
    )

    print(response.text)

except Exception as e:
    print(f"Erro na API Gemini: {e}")
