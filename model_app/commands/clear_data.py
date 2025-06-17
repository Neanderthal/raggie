import logging
from typing import Optional

from model_app.db.db import (
    clear_all_data,
    clear_user_data,
    clear_scope_data,
    clear_user_scope_data,
)

logger = logging.getLogger(__name__)


def clear_data(
    username: Optional[str] = None,
    scope_name: Optional[str] = None,
    confirm: bool = False
) -> None:
    """
    Clear RAG data from the database based on provided filters.
    
    Args:
        username: If provided, clear data only for this user
        scope_name: If provided, clear data only for this scope
        confirm: If True, skip confirmation prompt
    """
    
    # Determine what will be cleared
    if username and scope_name:
        action_desc = f"data for user '{username}' in scope '{scope_name}'"
        clear_func = lambda: clear_user_scope_data(username, scope_name)
    elif username:
        action_desc = f"all data for user '{username}'"
        clear_func = lambda: clear_user_data(username)
    elif scope_name:
        action_desc = f"all data in scope '{scope_name}'"
        clear_func = lambda: clear_scope_data(scope_name)
    else:
        action_desc = "ALL RAG data from the database"
        clear_func = clear_all_data
    
    # Confirmation prompt
    if not confirm:
        response = input(f"Are you sure you want to clear {action_desc}? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Operation cancelled.")
            return
    
    try:
        deleted_count = clear_func()
        print(f"Successfully cleared {action_desc}.")
        print(f"Deleted {deleted_count} documents.")
        
        if deleted_count == 0:
            if username and scope_name:
                print(f"No data found for user '{username}' in scope '{scope_name}'.")
            elif username:
                print(f"No data found for user '{username}'.")
            elif scope_name:
                print(f"No data found for scope '{scope_name}'.")
            else:
                print("No data found in the database.")
                
    except Exception as e:
        logger.error(f"Error clearing data: {str(e)}")
        print(f"Error clearing data: {str(e)}")
        raise
