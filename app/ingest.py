import json
import os
import faiss
import numpy as np
from fastembed import TextEmbedding

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DATA_PATH = "data/politicas_parceiros.md"
VECTORSTORE_DIR = "vectorstore"

def load_and_chunk_markdown(file_path: str) -> list[dict]:
    """Lê o arquivo Markdown e divide por seções (##) mantendo metadados."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Divide o texto sempre que encontrar um subtitulo '## '
    raw_sections = content.split("\n## ")
    chunks = []

    for i, section in enumerate(raw_sections):
        text = section.strip()
        if not text:
            continue

        # Ignora o título principal (# ...) se ele estiver sozinho no primeiro bloco
        if i == 0 and text.startswith("# ") and "\n" not in text:
            continue

        # Restaura o '## ' nos blocos divididos e extrai o título da regra
        if i > 0:
            text = f"## {text}"

        lines = text.splitlines()
        title = lines[0].replace("#", "").strip()

        chunks.append(
            {
                "id": len(chunks),
                "source": os.path.basename(file_path),
                "section": title,
                "content": text,
            }
        )
    return chunks

def build_vectorstore():
    print("1. Lendo e fatiando documentos (Chunking)...")
    chunks = load_and_chunk_markdown(DATA_PATH)

    if not chunks:
        print("Erro: O arquivo Markdown está vazio ou não gerou trechos válidos. Ingestão abortada.")
        return

    print(f"   -> {len(chunks)} trechos extraídos com sucesso.")

    print(f"2. Carregando modelo Hugging Face ({EMBEDDING_MODEL})...")

    #Na primeira execução, ele baixa o modelo (~120 MB) e salva em cache
    embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)

    print("3. Gerando embeddings (vetores semânticos)...")
    texts = [chunk["content"] for chunk in chunks]
    embeddings_list = list(embedding_model.embed(texts))
    embeddings_matrix = np.array(embeddings_list, dtype="float32")

    # Normaliza os vetores para usar similaridade de cosseno (padrão em buscas semânticas)
    faiss.normalize_L2(embeddings_matrix)

    print("4. Criando e salvando índice FAISS...")
    dimension = embeddings_matrix.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner Product + L2 norm = Cosine Similarity
    index.add(embeddings_matrix)

    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(VECTORSTORE_DIR, "index.faiss"))

    with open(
        os.path.join(VECTORSTORE_DIR, "chunks.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print("Ingestão concluída! Pasta 'vectorstore/' gerada com sucesso.")


if __name__ == "__main__":
    build_vectorstore()