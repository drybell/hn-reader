from pydantic import (
    BaseModel
    , UUID4
    , AnyUrl
    , PostgresDsn
    , computed_field
)
from pydantic_settings import BaseSettings, SettingsConfigDict
from ipaddress import IPv4Address as IPv4

class ServerConfig(BaseModel):
    host : IPv4
    port : int

class DBConfig(BaseModel):
    username : str
    password : str
    database : str
    host     : str
    port     : int

    @computed_field
    @property
    def postgres_db_uri(self) -> PostgresDsn:
        return PostgresDsn.build(
            scheme="postgresql+psycopg",
            username=self.username,
            password=self.password,
            host=self.host,
            port=self.port,
            path=self.database,
        )

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env'
        , env_nested_delimiter="_"
        , env_prefix="BASE_"
    )

    server : ServerConfig
    db     : DBConfig
