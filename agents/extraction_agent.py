import sys
import os
import re
import json
import pandas as pd
from sqlalchemy import text

# ── Import correct pour les nouvelles versions de langchain ──
try:
    from langchain_ollama import ChatOllama
    from langchain_core.messages import SystemMessage, HumanMessage
    OLLAMA_OK = True
except ImportError:
    OLLAMA_OK = False

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
        self.engine  = self.connect_to_db()

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
            raise ValueError(f"Type de BDD non supporté : {self.db_type}")

    def extract_data(self, query):
        with self.engine.connect() as connection:
            result = connection.execute(text(query))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
        return df

    def extract_table(self, table_name):
        return self.extract_data(f"SELECT * FROM {table_name}")

    def extract_with_conditions(self, table_name, conditions):
        return self.extract_data(f"SELECT * FROM {table_name} WHERE {conditions}")

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
        os.makedirs(file_path, exist_ok=True)
        for table in tables['TABLE_NAME']:
            self.extract_table_to_csv(table, f"{file_path}/{table}.csv")
        return f"All tables extracted to {file_path}"

    # ── NOUVELLE VERSION ROBUSTE ──
    def extract_with_ollama_to_csv(self, file_path):

        # Vérifier si Ollama est disponible
        if not OLLAMA_OK:
            raise RuntimeError(
                "langchain_ollama non installé. "
                "Exécute : pip install langchain-ollama langchain-core"
            )

        # 1. Récupérer la liste des tables
        tables_df  = self.get_all_tables()
        table_list = tables_df['TABLE_NAME'].tolist()

        if not table_list:
            raise ValueError("Aucune table trouvée dans la base de données.")

        # 2. Appeler le LLM
        try:
            model = ChatOllama(model="llama2", temperature=0)

            messages = [
                SystemMessage(content="""
You are a data quality expert.
Your task is to analyze database table names and decide which ones may require data cleaning.
Return ONLY a valid Python list of table names, nothing else.
Example of valid response: ["customers", "orders"]
Do not add any explanation, just the list.
"""),
                HumanMessage(
                    content=f"The database contains these tables: {table_list}. "
                            f"Which tables should be cleaned? Return only a Python list."
                )
            ]

            response = model.invoke(messages)
            raw_text = response.content.strip()

        except Exception as e:
            raise RuntimeError(f"Erreur Ollama : {str(e)}. Vérifiez qu'Ollama est lancé (ollama serve).")

        # 3. Parser la réponse du LLM de façon sécurisée
        tables_to_clean = self._parse_llm_list(raw_text, table_list)

        # 4. Extraire les tables choisies
        os.makedirs(file_path, exist_ok=True)
        extracted = []

        for table in tables_to_clean:
            # Vérifier que la table existe vraiment dans la BDD
            if table in table_list:
                df = self.extract_table(table)
                df.to_csv(f"{file_path}/{table}.csv", index=False)
                extracted.append(table)
            else:
                print(f"[WARN] Table '{table}' suggérée par le LLM mais introuvable dans la BDD.")

        if not extracted:
            # Fallback : extraire toutes les tables si le LLM n'a rien retourné de valide
            print("[WARN] LLM n'a retourné aucune table valide. Extraction de toutes les tables.")
            for table in table_list:
                df = self.extract_table(table)
                df.to_csv(f"{file_path}/{table}.csv", index=False)
                extracted.append(table)

        return {
            "tables_in_db":      table_list,
            "tables_chosen":     extracted,
            "llm_raw_response":  raw_text,
        }

    def _parse_llm_list(self, text, fallback_list):
        """
        Parse la réponse du LLM pour extraire une liste de tables.
        Essaie plusieurs méthodes pour être robuste.
        """
        # Méthode 1 : chercher une liste Python dans le texte [ ... ]
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group().replace("'", '"'))
                if isinstance(result, list) and len(result) > 0:
                    return result
            except Exception:
                pass

        # Méthode 2 : chercher les noms de tables connus dans la réponse
        found = []
        for table in fallback_list:
            if table.lower() in text.lower():
                found.append(table)
        if found:
            return found

        # Méthode 3 : fallback → retourner toutes les tables
        return fallback_list