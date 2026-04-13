from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    gateway_secret: str
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    huggingface_token: str

    class Config:
        env_file = ".env"

settings = Settings()