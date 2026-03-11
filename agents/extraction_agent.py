import sys
import os
import pandas as pd
from sqlalchemy import text
from langchain_ollama import ChatOllama
from langchain.messages import SystemMessage, HumanMessage
 
# ajouter la racine du projet au path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from db.connection import (
    connect_mysql,
    connect_postgresql,
    connect_sqlite,
    connect_oracle,
    connect_sql_server
)


class ExtractionAgent:

    def __init__(self, db_type, db_name):
        self.db_type = db_type
        self.db_name = db_name
        self.engine = self.connect_to_db()


    def connect_to_db(self):

        if self.db_type == "mysql":
            return connect_mysql(self.db_name)

        elif self.db_type == "postgresql":
            return connect_postgresql(self.db_name)

        elif self.db_type == "sqlite":
            return connect_sqlite(self.db_name)

        elif self.db_type == "oracle":
            return connect_oracle(self.db_name)

        elif self.db_type == "sql_server":
            return connect_sql_server(self.db_name)

        else:
            raise ValueError("Unsupported database type")

    # exécuter une requête SQL
    def extract_data(self, query):

        with self.engine.connect() as connection:

            result = connection.execute(text(query))

            df = pd.DataFrame(result.fetchall(), columns=result.keys())

        return df

    # extraire toute une table
    def extract_table(self, table_name):

        query = f"SELECT * FROM {table_name}"

        return self.extract_data(query)

    # extraire avec condition
    def extract_with_conditions(self, table_name, conditions):

        query = f"SELECT * FROM {table_name} WHERE {conditions}"

        return self.extract_data(query)

    # exporter une table en CSV
    def extract_table_to_csv(self, table_name, file_path):

        df = self.extract_table(table_name)

        df.to_csv(file_path, index=False)

        return f"Data from {table_name} extracted to {file_path}"

  
    def extract_columns_name(self, table_name):

        query = f"""
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = '{table_name}'
        """

        return self.extract_data(query)

    
    def extract_table_schema(self, table_name):

        query = f"""
        SELECT COLUMN_NAME, DATA_TYPE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = '{table_name}'
        """

        return self.extract_data(query)
    
    def get_all_tables(self):

        query = """
        SELECT TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_TYPE = 'BASE TABLE'
        """
        return self.extract_data(query)
    
    def extract_all_data_to_csv(self, file_path):

        tables = self.get_all_tables()

        for table in tables['TABLE_NAME']:
            self.extract_table_to_csv(table, f"{file_path}/{table}.csv")

        return f"All tables extracted to {file_path}"
    
    

    def extract_all_data_to_csv(self, file_path):

        tables = self.get_all_tables()

        os.makedirs(file_path, exist_ok=True)

        for table in tables['TABLE_NAME']:
            self.extract_table_to_csv(table, f"{file_path}/{table}.csv")

        return f"All tables extracted to {file_path}"

    # extraction intelligente avec LLM
    def extract_with_ollama_to_csv(self, file_path):

        tables = self.get_all_tables()
        table_list = tables['TABLE_NAME'].tolist()

        model = ChatOllama(model="llama2", temperature=0)

        messages = [
            SystemMessage(
                content="""
You are a data quality expert.
Your task is to analyze database tables and decide which ones may require data cleaning.
Return ONLY a Python list of table names.
"""
            ),
            HumanMessage(
                content=f"The database contains these tables: {table_list}. Which tables should be cleaned?"
            )
        ]

        response = model.invoke(messages)

        tables_to_clean = eval(response.content)

        os.makedirs(file_path, exist_ok=True)

        for table in tables_to_clean:

            df = self.extract_table(table)

            df.to_csv(f"{file_path}/{table}.csv", index=False)

        return f"Tables extracted: {tables_to_clean}"






    