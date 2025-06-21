from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import DirectoryPath


class ModelSettings(BaseSettings):
    """
    ML Model Configuration settings for the application

    Attributes:
        model_config (SettingsconfigDict): Model config, loaded from .env file
        model_path (DirectoryPath): FileSystem path to the model
        model_name (str): Name of the ML model
    """

    model_config = SettingsConfigDict(env_file='config/.env',
                                      env_file_encoding='utf-8',
                                      extra='ignore')

    model_path: DirectoryPath
    model_name: str


model_settings = ModelSettings()
