from sqlalchemy import create_engine, Column, TIMESTAMP, String, Float, PrimaryKeyConstraint, Boolean, JSON
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from  functions.data_utils import get_db_url
import numpy as np
from datetime import datetime
import pandas as pd

# Database URL
# Update this with the correct database URL if needed
db_url = get_db_url()
engine = create_engine(db_url)

# Base declarative class
Base = declarative_base()


def convert_numpy_types(obj):
    """
    Recursively convert NumPy types to native Python types.
    Handles nested dicts, lists, and common numpy types.
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.to_pydatetime()
    else:
        return obj


# AnalysisDataClass definition
class AnalysisDataClass(Base):
    __tablename__ = 'analysis_data'

    date = Column(TIMESTAMP, nullable=False)
    interval = Column(String, nullable=False)
    buy_indicator = Column(String)
    analysis_symbol = Column(String, nullable=False)
    analysis_last_close_price = Column(Float)
    analysis_token = Column(String)
    application_status = Column(JSON)
    type_one = Column(Boolean, default=False)
    nifty = Column(Boolean, default=False)
    higher_interval_divergence = Column(Boolean, default=False)
    stoploss = Column(Float, default=False)
    volume = Column(Float)
    target = Column(Float)
    order_params = Column(JSONB)
    __table_args__ = (
        PrimaryKeyConstraint('date', 'interval', 'analysis_symbol', name='analysis_data_pk'),
    )


# Function to upload analysis data
def upload_analysis_data(interval, analysis_dict):
    """
    Uploads analysis data to the database. Updates existing entries or inserts new ones.

    Args:
        interval (str): The interval for the analysis.
        analysis_dict (dict): The analysis data dictionary containing the required fields.
    """
    analysis_dict['interval'] = interval
    
    # Convert numpy types to native Python types
    analysis_dict = convert_numpy_types(analysis_dict)
    
    Session = sessionmaker(bind=engine)
    session = Session()

    # Ensure table exists
    Base.metadata.create_all(engine)
    try:
        # Check if the record already exists
        existing_entry = session.query(AnalysisDataClass).filter_by(
            date=analysis_dict['date'],
            interval=analysis_dict['interval'],
            analysis_symbol=analysis_dict['analysis_symbol']
        ).first()

        if existing_entry:
            # Update the existing entry
            for key, value in analysis_dict.items():
                setattr(existing_entry, key, value)
        else:
            # Insert new entry
            new_entry = AnalysisDataClass(**analysis_dict)
            session.add(new_entry)

        # Commit the transaction
        session.commit()
    except Exception as e:
        # Rollback in case of an error
        session.rollback()
        raise e
    finally:
        # Close the session
        session.close()
