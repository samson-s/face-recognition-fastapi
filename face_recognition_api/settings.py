from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    QDRANT_MODE: str = "local"  # local or remote
    QDRANT_PATH: str = "qdrant_db/"
    QDRANT_HOST: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION: str = "face_encodings"
    QDRANT_PORT: int = 6333


@lru_cache()
def get_settings():
    return Settings()
