from sqlalchemy import create_engine, Column, TEXT, NUMERIC, TIMESTAMP, ARRAY, String, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
from functions.data_utils import get_db_url

engine = create_engine(get_db_url())
Base = declarative_base()


class IDictClass(Base):
    __tablename__ = 'i_dict'

    analysis_symbol = Column(TEXT, primary_key=True)
    interval = Column(TEXT)
    target_exit_price = Column(NUMERIC)
    time_stoploss = Column(TIMESTAMP)
    buy_indicator = Column(TEXT)
    analysis_buy_price = Column(NUMERIC)
    analysis_profit_percentage = Column(NUMERIC)
    user_id_list = Column(ARRAY(String))
    analysis_symbol_token = Column(TEXT)
    mod_list = Column(ARRAY(String))
    buy_datetime = Column(TIMESTAMP)
    exit_position = Column(JSON)
    stoploss_exit_price = Column(NUMERIC)


def upload_i_dict(i_dict):
    """Insert or update an entry in the i_dict table."""
    Session = sessionmaker(bind=engine)
    session = Session()
    Base.metadata.create_all(engine)

    try:
        existing_record = (
            session.query(IDictClass)
            .filter_by(analysis_symbol=i_dict['analysis_symbol'], interval=i_dict.get('interval'))
            .first()
        )

        if existing_record:
            for key, value in i_dict.items():
                if hasattr(existing_record, key):
                    setattr(existing_record, key, value)
        else:
            new_payload = {column.name: i_dict.get(column.name) for column in IDictClass.__table__.columns}
            new_record = IDictClass(**new_payload)
            session.add(new_record)

        session.commit()
    except IntegrityError as exc:
        session.rollback()
        print(f"Error: {exc}")
    finally:
        session.close()
