import sys
import os
import pandas as pd
from sqlalchemy import text

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

    # connexion à la base
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

    # récupérer les noms des colonnes
    def extract_columns_name(self, table_name):

        query = f"""
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = '{table_name}'
        """

        return self.extract_data(query)

    # récupérer le schema de la table
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