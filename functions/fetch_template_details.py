"""
Fetch template details for a user from all_templates_dict table.
Provides per-user trading configuration from the database.
"""
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from functions.data_utils import get_db_url
import logging

logger = logging.getLogger(__name__)

# Connection pooling
engine = create_engine(
    get_db_url(),
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_recycle=3600
)


def fetch_template_details(user_id, template_id=None):
    """
    Fetch template details for a user from all_templates_dict.
    
    Args:
        user_id: User ID to fetch template for
        template_id: Optional specific template ID (uses first if not provided)
    
    Returns:
        dict: Template details with normalized fields, or None if not found
    """
    try:
        if template_id:
            query = text("""
                SELECT *
                FROM all_templates_dict
                WHERE user_id = :user_id AND template_id = :template_id
            """)
            params = {'user_id': user_id, 'template_id': template_id}
        else:
            # Get first/default template for user
            query = text("""
                SELECT *
                FROM all_templates_dict
                WHERE user_id = :user_id
                LIMIT 1
            """)
            params = {'user_id': user_id}

        with engine.connect() as connection:
            row = connection.execute(query, params).mappings().first()

        if not row:
            logger.debug(f"No template found for user {user_id}")
            return None

        result_dict = dict(row)

        # Normalize selected_sectors
        selected_sectors = result_dict.get('selected_sectors')
        if isinstance(selected_sectors, (tuple, list)):
            result_dict['selected_sectors'] = list(selected_sectors)
        elif selected_sectors is None:
            result_dict['selected_sectors'] = []

        # Normalize sell percentages
        sell_percentage = result_dict.get('sell_percentage')
        if sell_percentage is None:
            sell_percentage = result_dict.get('target_sell_percentage')
        result_dict['sell_percentage'] = sell_percentage

        # Normalize target/stoploss types
        target_type = result_dict.get('target_sell_percentage_type')
        stoploss_type = result_dict.get('stoploss_sell_percentage_type')

        if target_type is None:
            target_type = result_dict.get('sell_percentage_type')
            result_dict['target_sell_percentage_type'] = target_type
        if stoploss_type is None:
            stoploss_type = result_dict.get('sell_percentage_type', target_type)
            result_dict['stoploss_sell_percentage_type'] = stoploss_type

        if 'sell_percentage_type' not in result_dict or result_dict['sell_percentage_type'] is None:
            result_dict['sell_percentage_type'] = target_type

        # Normalize intervals
        intervals = result_dict.get('intervals')
        if isinstance(intervals, (tuple, list)):
            result_dict['intervals'] = list(intervals)
        elif intervals is None:
            result_dict['intervals'] = []

        # Provide defaults for key fields
        result_dict.setdefault('segment', 'Options')
        result_dict.setdefault('mode', 'Custom')
        result_dict.setdefault('withdraw_amount', 0.0)
        result_dict.setdefault('maximum_buy_percentage', None)
        result_dict.setdefault('target_exit_percentage', None)
        result_dict.setdefault('stoploss_exit_percentage', None)
        result_dict.setdefault('selected_mod', None)
        result_dict.setdefault('entry_mod', None)   # NEW: Entry moderator
        result_dict.setdefault('exit_mod', None)    # NEW: Exit moderator

        return result_dict

    except Exception as exc:
        logger.error(f"Error fetching template for user {user_id}: {exc}")
        return None


def get_all_active_templates():
    """
    Fetch all active user templates for batch processing.
    
    Returns:
        list: List of template dictionaries for all users
    """
    try:
        query = text("""
            SELECT DISTINCT ON (user_id) *
            FROM all_templates_dict
            ORDER BY user_id, template_id
        """)

        with engine.connect() as connection:
            rows = connection.execute(query).mappings().all()

        templates = []
        for row in rows:
            template = fetch_template_details(row['user_id'], row.get('template_id'))
            if template:
                templates.append(template)

        logger.info(f"Fetched {len(templates)} active templates")
        return templates

    except Exception as exc:
        logger.error(f"Error fetching all templates: {exc}")
        return []
