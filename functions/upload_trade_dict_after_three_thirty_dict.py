from sqlalchemy import create_engine, Column, TIMESTAMP, String, Float, Integer, Numeric
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
from functions.data_utils import get_db_url
# TimescaleDB connection details
engine = create_engine(get_db_url())

Base = declarative_base()


class TradeDataClass(Base):
    __tablename__ = 'trade_data'

    user_id = Column(String)
    analysis_symbol = Column(String)
    interval = Column(String)
    order_id = Column(String, primary_key=True)
    order_variety = Column(String)
    tradingsymbol = Column(String)
    symbol_token = Column(String)
    transaction_type = Column(String)
    start_datetime = Column(TIMESTAMP)
    exchange = Column(String)
    product_type = Column(String)
    duration = Column(String)
    quantity = Column(Integer)
    trade_mode = Column(String)
    buy_price = Column(Numeric)
    order_status = Column(String)
    end_datetime = Column(TIMESTAMP)
    sell_price = Column(Numeric)
    sell_mode = Column(String)


class AfterThreeThirtyClass(Base):
    __tablename__ = 'after_three_thirty_data'
    user_id = Column(String)
    analysis_symbol = Column(String, nullable=False)
    interval = Column(String, nullable=False)
    order_id = Column(String, primary_key=True)
    tradingsymbol = Column(String)
    product_type = Column(String)
    quantity = Column(Integer)
    exchange = Column(String)
    symbol_token = Column(String)
    transaction_type = Column(String)
    start_datetime = Column(TIMESTAMP)
    analysis_profit_percentage = Column(Float)
    buy_indicator = Column(String)
    trade_mode = Column(String)
    buy_price = Column(Numeric)
    order_status = Column(String)
    sector = Column(String)
    end_datetime = Column(TIMESTAMP)
    sell_price = Column(Numeric)
    sell_mode = Column(String)


def upload_tradedict_afterthreethirty(input_dictionary, filename):
    Session = sessionmaker(bind=engine)
    session = Session()
    Base.metadata.create_all(engine)

    try:
        if filename == 'trade_data':
            # Check if the record exists
            existing_record = session.query(TradeDataClass).filter_by(order_id=input_dictionary['order_id']).first()

            if existing_record:
                # Update existing record with fields in input_dictionary
                for key, value in input_dictionary.items():
                    setattr(existing_record, key, value)
            else:
                # Insert new record
                new_trade_data = TradeDataClass(**input_dictionary)
                session.add(new_trade_data)

        else:
            # Check if the record exists for AfterThreeThirtyClass
            existing_record = session.query(AfterThreeThirtyClass).filter_by(
                order_id=input_dictionary['order_id']).first()

            if existing_record:
                # Update existing record with fields in input_dictionary
                for key, value in input_dictionary.items():
                    setattr(existing_record, key, value)
            else:
                # Insert new record
                new_trade_data = AfterThreeThirtyClass(**input_dictionary)
                session.add(new_trade_data)

        session.commit()
    except IntegrityError as e:
        session.rollback()
        print(f"Error: {e}")
    finally:
        session.close()


def is_successful_trade_in_trade_data(user_id, analysis_symbol):
    """
    Checks if there is a successful trade (order_status 'BOUGHT') for the given user_id and analysis_symbol
    in both the trade_data and after_three_thirty_data tables.

    Returns:
        True if a record exists in both tables; otherwise, False.
    """
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        # Query for successful trade from the trade_data table
        trade_data_record = session.query(TradeDataClass).filter_by(
            user_id=user_id,
            analysis_symbol=analysis_symbol,
            order_status='BOUGHT'
        ).first()

        # Query for successful trade from the after_three_thirty_data table
        after_three_record = session.query(AfterThreeThirtyClass).filter_by(
            user_id=user_id,
            analysis_symbol=analysis_symbol,
            order_status='BOUGHT'
        ).first()

        # Explicit check: return True if both records exist; otherwise, return False.
        if trade_data_record is not None and after_three_record is not None:
            return True
        else:
            return False
    finally:
        session.close()
