from dotenv import load_dotenv
from sqlalchemy import text

load_dotenv()

from ekb.base import sql_interface as sqli

if __name__ == '__main__':
    sqli.create_all()