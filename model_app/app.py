import argparse
import asyncio
import os
import sys
from enum import Enum
from dotenv import find_dotenv, load_dotenv

# Add the parent directory to sys.path to make model_app importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model_app.commands.chat import chat
from model_app.commands.create_db import create_db
from model_app.commands.import_data import import_data
from model_app.commands.clear_data import clear_data

load_dotenv(find_dotenv(".env"))


class Command(Enum):
    CREATE_DB = "create-db"
    IMPORT_DATA = "import-data"
    CHAT = "chat"
    CLEAR_DATA = "clear-data"


async def main():
    parser = argparse.ArgumentParser(description="Application Description")

    subparsers = parser.add_subparsers(
        title="Subcommands",
        dest="command",
        help="Display available subcommands",
    )

    # create-db command
    create_db_parser = subparsers.add_parser(
        Command.CREATE_DB.value, help="Create a database"
    )
    create_db_parser.set_defaults(func=create_db)

    # import-data command
    import_data_parser = subparsers.add_parser(
        Command.IMPORT_DATA.value, help="Import data with user and scope"
    )
    import_data_parser.add_argument(
        "data_source",
        type=str,
        help="Specify the document data source (.pdf, .md, .docx)",
    )
    import_data_parser.add_argument(
        "--username", type=str, required=True, help="Username for data ownership"
    )
    import_data_parser.add_argument(
        "--scope", type=str, required=True, help="Scope for data organization"
    )
    import_data_parser.add_argument(
        "--document-name", type=str, required=False, help="Optional document name (defaults to filename)"
    )
    import_data_parser.set_defaults(func=import_data)

    # chat command
    chat_parser = subparsers.add_parser(Command.CHAT.value, help="Use chat feature")
    chat_parser.add_argument(
        "--username",
        type=str,
        required=False,
        help="Username for filtering chat history",
        default="",
    )
    chat_parser.add_argument(
        "--scope",
        type=str,
        required=False,
        help="Scope for filtering chat history",
        default="",
    )
    chat_parser.set_defaults(func=chat)

    # clear-data command
    clear_data_parser = subparsers.add_parser(
        Command.CLEAR_DATA.value, help="Clear RAG data from database"
    )
    clear_data_parser.add_argument(
        "--username",
        type=str,
        required=False,
        help="Clear data only for this user",
    )
    clear_data_parser.add_argument(
        "--scope",
        type=str,
        required=False,
        help="Clear data only for this scope",
    )
    clear_data_parser.add_argument(
        "--confirm",
        action="store_true",
        help="Skip confirmation prompt",
    )
    clear_data_parser.set_defaults(func=clear_data)

    args = parser.parse_args()

    if hasattr(args, "func"):
        if args.command == Command.CREATE_DB.value:
            args.func()
        elif args.command == Command.CHAT.value:
            await args.func(username=args.username, scope_name=args.scope)
        elif args.command == Command.IMPORT_DATA.value:
            args.func(args.data_source, username=args.username, scope_name=args.scope, document_name=args.document_name)
        elif args.command == Command.CLEAR_DATA.value:
            args.func(username=args.username, scope_name=args.scope, confirm=args.confirm)
    else:
        print("Invalid command. Use '--help' for assistance.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Fatal error: {str(e)}", file=sys.stderr)
        sys.exit(1)
