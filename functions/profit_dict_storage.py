"""Storage functions for profit dict."""
import json
import numpy as np
import pytz
from datetime import datetime
from sqlalchemy import Column, String, TIMESTAMP, create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import JSONB
from functions.data_utils import get_db_url

Base = declarative_base()
engine = create_engine(get_db_url(), connect_args={'sslmode': 'require'})


class ProfitDict(Base):
    __tablename__ = 'profitdictnew'
    symbol = Column(String, primary_key=True)
    interval = Column(String, primary_key=True)
    data = Column(JSONB, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True))


def convert_numpy(obj):
    """Convert numpy types to Python types."""
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, np.int64):
        return int(obj)
    return obj


def merge_trade_histories(existing, new, update=False):
    """Merge trade histories by (from, to) key."""
    trades = {(t['from'], t['to']): t for t in existing}
    for t in new:
        key = (t['from'], t['to'])
        if key not in trades:
            trades[key] = t
        elif update:
            trades[key].update(t)
    return sorted(trades.values(), key=lambda x: x['from'])


def get_existing_symbol_history(symbol, interval):
    """Get existing trades from database."""
    try:
        query = text("SELECT data FROM profitdictnew WHERE symbol = :s AND interval = :i")
        with engine.connect() as conn:
            result = conn.execute(query, {'s': symbol, 'i': interval}).fetchone()
            if result:
                data = result[0]
                if isinstance(data, str):
                    data = json.loads(data)
                return data.get('symbol_history', [])
    except:
        pass
    return []


def save_profit_dict_row(symbol, interval, data):
    """Save profit dict to database with proper batch_results handling."""
    try:
        Session = sessionmaker(bind=engine)
        session = Session()
        existing = session.query(ProfitDict).filter_by(symbol=symbol, interval=interval).first()
        data = convert_numpy(data)
        
        if existing:
            existing_data = existing.data if isinstance(existing.data, dict) else json.loads(existing.data)
            
            # Merge symbol histories
            merged_history = merge_trade_histories(
                existing_data.get('symbol_history', []), 
                data.get('symbol_history', [])
            )
            
            # Build updated data matching original structure
            updated_data = {
                'from_date': data.get('from_date'),
                'to_date': data.get('to_date'),
                'batch_results': data.get('batch_results', []),
                'total_trades': len(merged_history),
                'symbol_history': merged_history
            }
            updated_data = convert_numpy(updated_data)
            
            existing.data = updated_data
            existing.created_at = datetime.now(pytz.timezone('Asia/Kolkata'))
        else:
            session.add(ProfitDict(symbol=symbol, interval=interval, data=data,
                                   created_at=datetime.now(pytz.timezone('Asia/Kolkata'))))
        session.commit()
        return True
    except Exception as e:
        print(f"Save error: {e}")
        return False
