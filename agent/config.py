import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")


@dataclass(frozen=True)
class Settings:
    model_name: str
    model_api_key: str
    model_base_url: str
    model_thinking_type: str
    youtube_api_key: str

    @property
    def has_model_api_key(self) -> bool:
        return bool(self.model_api_key)

    @property
    def has_youtube_key(self) -> bool:
        return bool(self.youtube_api_key)


def load_settings() -> Settings:
    return Settings(
        model_name=os.getenv("MODEL_NAME", "glm-4.5-air"),
        model_api_key=(
            os.getenv("MODEL_API_KEY")
            or os.getenv("ZAI_API_KEY")
            or os.getenv("OPENAI_API_KEY", "")
        ),
        model_base_url=os.getenv("MODEL_BASE_URL", "https://api.z.ai/api/paas/v4/"),
        model_thinking_type=os.getenv("MODEL_THINKING_TYPE", "enabled"),
        youtube_api_key=os.getenv("YOUTUBE_API_KEY", ""),
    )
