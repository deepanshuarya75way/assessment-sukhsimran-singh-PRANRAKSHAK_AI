from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Groq
    groq_api_key: str
    llm_model: str = "llama-3.3-70b-versatile"

    # Database
    database_url: str = "./pranrakshak.db"

    # Model
    model_path: str = "../model/saved_model.pkl"
    shap_explainer_path: str = "../model/shap_explainer.pkl"

    # RAG
    guidelines_path: str = "./guidelines.txt"

    # CORS
    frontend_origin: str = "http://localhost:5173"

    # Env
    env: str = "development"
    model_version: str = "1.0.0"
    feedback_api_key: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()