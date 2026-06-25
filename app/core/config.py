from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    APP_NAME: str = "Agentic Graph RAG"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # Folder where generated PNGs are saved and served via /static/
    STATIC_DIR: str = "static"

    # Paths where the FAISS vector store is saved after /retrieval
    FAISS_INDEX_PATH: str = "static/vector_store.faiss"
    FAISS_METADATA_PATH: str = "static/vector_store_metadata.json"

    # Path to your corpus — loaded automatically on startup, no upload needed
    CORPUS_PATH: str = "data/data_final_cleaned.txt"
    COLAB_PACKAGE_PATH: str = "static/colab_graph_rag_package.zip"

    # Neo4j Aura connection. Put real values in .env.
    NEO4J_URI: str = ""
    NEO4J_USERNAME: str = "neo4j"
    NEO4J_PASSWORD: str = ""
    NEO4J_DATABASE: str = "neo4j"
    RL_POLICY_PATH: str = "static/rl_policy.json"
    AGENTIC_CACHE_PATH: str = "static/agentic_response_cache.json"
    ENABLE_LLM_SYNTHESIS: bool = True
    LLM_PROVIDER: str = "local"
    LLM_MODEL_NAME: str = "mistralai/Mistral-7B-Instruct-v0.3"
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL_NAME: str = "gemini-2.5-flash"
    GEMINI_MAX_OUTPUT_TOKENS: int = 512
    GEMINI_RETRY_MAX_OUTPUT_TOKENS: int = 384
    GEMINI_THINKING_BUDGET: int = 0
    LLM_MAX_INPUT_TOKENS: int = 1536
    LLM_MAX_NEW_TOKENS: int = 140
    LLM_RETRY_MAX_NEW_TOKENS: int = 90
    LLM_GENERATION_MAX_TIME_SECONDS: float = 90.0
    LLM_ALLOW_CPU_GENERATION: bool = False

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()

# Make sure the static folder exists on startup
Path(settings.STATIC_DIR).mkdir(parents=True, exist_ok=True)
