
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client()

print("\nModelos disponíveis para geração de conteúdo:\n")

try:
    for model in client.models.list():
        if "generateContent" in (model.supported_actions or []):
            print(model.name)

except Exception as e:
    print(f"Erro ao consultar os modelos: {e}")
