from sqlalchemy import create_engine, Column, Integer, String, Sequence
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, Sequence('user_id_seq'), primary_key=True)
    name = Column(String(50))
    email = Column(String(50))

class AnalysisResult(Base):
    __tablename__ = 'analysis_results'
    id = Column(Integer, Sequence('result_id_seq'), primary_key=True)
    user_id = Column(Integer)
    result_data = Column(String)

def get_database_engine(db_url):
    return create_engine(db_url)

def create_tables(engine):
    Base.metadata.create_all(engine)

def get_session(engine):
    Session = sessionmaker(bind=engine)
    return Session()