import logging
import pytz
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import create_engine, Column, String, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from functions.data_utils import get_db_url

# Define your SQLAlchemy Base and candle_data table structure
Base = declarative_base()


class CandleData(Base):
    __tablename__ = 'candle_data'
    symbol = Column(String, primary_key=True)
    interval = Column(String, primary_key=True)
    datetime = Column(DateTime, primary_key=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)

IST = pytz.timezone("Asia/Kolkata")


# Function to save or overwrite candle data
def save_candle_data(candle_data_list):
    # TimescaleDB connection details
    db_url = get_db_url()
    engine = create_engine(db_url, connect_args={'sslmode': 'require'})

    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Deduplicate by (symbol, interval, datetime) before insert to avoid ON CONFLICT issues
        unique_rows = {}
        for row in candle_data_list:
            symbol = row.get('symbol')
            interval = row.get('interval')
            dt = row.get('datetime')

            if symbol is None or interval is None or dt is None:
                logging.warning("Skipping candle row with missing primary key fields: %s", row)
                continue

            row_copy = row.copy()
            row_copy['datetime'] = dt
            print("Saving Candle Data")
            print(dt)
            print(row_copy['datetime'])
            key = (row_copy['symbol'], row_copy['interval'], row_copy['datetime'])
            print(key)
            unique_rows[key] = row_copy  # later rows overwrite earlier duplicates

        if not unique_rows:
            logging.info("No candle data to save after deduplication.")
            return

        stmt = insert(CandleData).values(list(unique_rows.values()))

        # Define update behavior on conflict to overwrite all fields except primary keys
        update_dict = {
            column.name: stmt.excluded[column.name]
            for column in CandleData.__table__.columns
            if column.name not in ('symbol', 'interval', 'datetime')  # Exclude primary keys
        }

        stmt = stmt.on_conflict_do_update(
            index_elements=['symbol', 'interval', 'datetime'],
            set_=update_dict
        )

        # Execute the statement
        session.execute(stmt)
        session.commit()
        print("Saved or updated candle data")
    finally:
        session.close()
        engine.dispose()
