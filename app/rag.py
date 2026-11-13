import json
import os
import faiss
import numpy as np
from fastembed import TextEmbedding
from google import genai
from dotenv import load_dotenv

# Carrega a variável GEMINI_API_KEY do arquivo .env
load_dotenv()

# Usamos o MESMO modelo que gerou a base para não ter distorção
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VECTORSTORE_DIR = "vectorstore"

print("Inicializando o motor de busca e a LLM...")

# 1. Carrega o Banco de Dados Vetorial (FAISS) e os Documentos (JSON)
try:
    index = faiss.read_index(os.path.join(VECTORSTORE_DIR, "index.faiss"))
    with open(os.path.join(VECTORSTORE_DIR, "chunks.json"), "r", encoding="utf-8") as f:
        chunks = json.load(f)
except Exception as e:
    print("Erro: Pasta 'vectorstore' não encontrada. Rode o ingest.py primeiro!")
    exit(1)

# 2. Inicia o modelo local de embeddings e o cliente do Gemini
embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)
gemini_client = genai.Client()  # Ele puxa a chave do os.environ["GEMINI_API_KEY"] sozinho

def retrieve(query: str, top_k: int = 2) -> list[dict]:
    """Busca as regras mais parecidas com a pergunta do parceiro."""
    
    # 1. Verifica se o índice possui documentos suficientes para o top_k solicitado
    if index.ntotal == 0:
        return []
    safe_top_k = min(top_k, index.ntotal)

    # 2. Transforma a pergunta em vetor
    query_embedding = list(embedding_model.embed([query]))[0]
    query_vector = np.array([query_embedding], dtype="float32")
    
    # 3. Valida a compatibilidade dimensional do índice (Prevenção de quebra de contrato)
    if query_vector.shape[1] != index.d:
        raise ValueError(f"Dimensão incompatível: vetor da query tem {query_vector.shape[1]}, mas o índice espera {index.d}")
        
    faiss.normalize_L2(query_vector)

    # 4. Busca as 'safe_top_k' maiores similaridades (Produto Interno de vetores normalizados = Cosseno)
    similarities, indices = index.search(query_vector, safe_top_k)
    
    results = []
    for i in range(safe_top_k):
        idx = indices[0][i]
        # O FAISS retorna -1 para vagas não preenchidas se algo falhar na contagem
        if idx != -1:
            results.append(chunks[idx])
            
    return results

def ask_copilot(query: str) -> str:
    """Faz a busca semântica e orquestra a resposta com o Gemini."""
    # A. Recuperar contexto do FAISS
    relevant_chunks = retrieve(query)
    
    if not relevant_chunks:
        return "Não encontrei regras aplicáveis na base de conhecimento."

    # B. Montar o Contexto exato para a IA não inventar coisas (Zero Alucinação)
    context_text = "\n\n".join([f"[{c['section']}]\n{c['content']}" for c in relevant_chunks])
    
    prompt = f"""Você é um Copiloto de Suporte Técnico para Parceiros do iFood. 
    Sua missão é responder à dúvida do parceiro de forma educada, clara e direta.

    REGRA DE OURO: Baseie sua resposta APENAS nas políticas listadas no Contexto abaixo. 
    Se a resposta não estiver ali, diga educadamente que não possui essa informação.
    Sempre mencione o nome da regra que embasou sua resposta.

CONTEXTO (Políticas Oficiais):
{context_text}

PERGUNTA DO PARCEIRO:
{query}
"""

    # C. Chamar a API do Gemini
    response = gemini_client.models.generate_content(
        model='gemini-3.8-flash',
        contents=prompt,
    )
    
    return response.text

if __name__ == "__main__":
    print("\n✅ Copiloto iFood Pronto! (Digite 'sair' para encerrar)")
    while True:
        pergunta = input("\n👤 Parceiro: ")
        if pergunta.lower() in ['sair', 'exit', 'quit']:
            break
            
        print("⚙️  Buscando regras e consultando a IA...\n")
        resposta = ask_copilot(pergunta)
        print(f"🤖 Copiloto: {resposta}")