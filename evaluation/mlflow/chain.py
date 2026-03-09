from __future__ import annotations

import sys
import os
import mlflow
# Get the absolute path of the directory containing 'src'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
print(project_root)
# Add the project root to the system path
sys.path.insert(0, project_root)

# Now you can import from src




import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

from src.backend.retriever import get_ensemble_retriever



load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if not GOOGLE_API_KEY:
    raise RuntimeError("Falta GOOGLE_API_KEY en el .env")

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    api_key=GOOGLE_API_KEY,
    temperature=0.2,
    max_output_tokens=2048,
    convert_system_message_to_human=True
)

def _build_context_block(docs: List[Document]) -> str:
    """
    Convierte la lista de documentos en un bloque de contexto legible,
    incluyendo metadatos básicos (source, chunk_id).
    """
    bloques: List[str] = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        source = meta.get("source", "desconocido")
        chunk_id = meta.get("chunk_id", meta.get("id", f"doc_{i}"))
        bloque = (
            f"[doc{i} | source={source} | chunk_id={chunk_id}]\n"
            f"{d.page_content}"
        )
        bloques.append(bloque)

    return "\n\n".join(bloques)


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Eres un asistente experto que responde en español.\n"
                "Debes contestar la pregunta del usuario usando EXCLUSIVAMENTE "
                "la información del contexto proporcionado.\n\n"
                "Si la respuesta no se puede obtener del contexto, indícalo de forma explícita "
                "y, si es útil, sugiere qué información adicional se requeriría.\n\n"
            ),
        ),
        (
            "human",
            "CONTEXTO:\n{context}\n\nPREGUNTA:\n{question}\n\nResponde en español.",
        ),
    ]
)


# ---------------------------------------------------------------------
# Chain RAG
# ---------------------------------------------------------------------

def load_model(model_path=None):
    """
    input (str) -> retriever -> docs -> contexto
                 -> PROMPT -> LLM -> texto
    """
    retriever = get_ensemble_retriever(k=8)

    rag_chain = (
        {
            "context": retriever | RunnableLambda(_build_context_block),
            "question": RunnablePassthrough(),
        }
        | PROMPT
        | llm
        | StrOutputParser()  # convierte AIMessage -> str
    )
    return rag_chain


chain = load_model(mlflow.get_artifact_uri())

mlflow.models.set_model(chain)
