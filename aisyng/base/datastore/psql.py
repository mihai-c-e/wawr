import os

from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker

from aisyng.base.datastore.sqla import SQLAPersistenceInterface
from aisyng.base.embeddings import EmbeddingPool

_db_srv = os.environ["DATA_DB_SERVER"]
_db_usr = os.environ["DATA_DB_USR"]
_db_pwd = os.environ["DATA_DB_PWD"]
_db_port = os.environ["DATA_DB_PORT"]
_db = os.environ["DATA_DB"]
_con_str = f'postgresql+psycopg2://{_db_usr}:{_db_pwd}@{_db_srv}:{_db_port}/{_db}'
engine = create_engine(_con_str)
Session = scoped_session(sessionmaker(engine))


class PSQLPersistenceInterface(SQLAPersistenceInterface):
    def __init__(self, embedding_pool: EmbeddingPool):
        super().__init__(embedding_pool=embedding_pool, session_factory=Session)
