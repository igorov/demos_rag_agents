from decouple import config

OPENAI_API_KEY: str = config("OPENAI_API_KEY")
OPENAI_MODEL: str = config("OPENAI_MODEL")
OPENAI_EMBEDDING_MODEL: str = config("OPENAI_EMBEDDING_MODEL", default="text-embedding-3-small")
ASSISTANT_NAME: str = config("ASSISTANT_NAME", default="Asistente")

DATABASE_URL: str = config("DATABASE_URL")
HISTORY_LIMIT: int = config("HISTORY_LIMIT", default=10, cast=int)
RETRIEVAL_LIMIT: int = config("RETRIEVAL_LIMIT", default=5, cast=int)

QDRANT_URL: str | None = config("QDRANT_URL", default=None)
QDRANT_KEY: str | None = config("QDRANT_API_KEY", default=None)
QDRANT_COLLECTION_NAME: str = config("QDRANT_COLLECTION_NAME", default="documents")
