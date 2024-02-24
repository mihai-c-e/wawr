from dotenv import load_dotenv

load_dotenv()

from base.tools import sql_interface as sqli

if __name__ == '__main__':
    sqli.create_all()