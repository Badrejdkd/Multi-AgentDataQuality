# agents/extraction_agent.py - AJOUTER CES MÉTHODES

import sys
import os
import re
import json
import pandas as pd
from sqlalchemy import text

# ── Import pour Ollama ──
try:
    from langchain_ollama import ChatOllama
    from langchain_core.messages import SystemMessage, HumanMessage
    OLLAMA_OK = True
except ImportError:
    OLLAMA_OK = False

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from db.connection import (
    connect_mysql, connect_postgresql, connect_sqlite,
    connect_oracle, connect_sql_server
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

    # ============================================================
    #  NOUVELLES MÉTHODES POUR LLM AVEC PROMPTS INTELLIGENTS
    # ============================================================

    def _check_ollama(self):
        """Vérifie si Ollama est disponible"""
        if not OLLAMA_OK:
            raise RuntimeError(
                "langchain_ollama non installé. "
                "Exécute : pip install langchain-ollama langchain-core"
            )
        return True

    def _call_llm(self, system_prompt, user_prompt, model="llama2", temperature=0):
        """Appelle le LLM avec les prompts fournis"""
        self._check_ollama()
        
        try:
            model = ChatOllama(model=model, temperature=temperature)
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = model.invoke(messages)
            return response.content.strip()
        except Exception as e:
            raise RuntimeError(f"Erreur Ollama : {str(e)}. Vérifiez qu'Ollama est lancé (ollama serve)")

    def _parse_llm_list(self, text, fallback_list):
        """
        Parse la réponse du LLM pour extraire une liste.
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

        # Méthode 2 : chercher les éléments connus dans la réponse
        found = []
        for item in fallback_list:
            if str(item).lower() in text.lower():
                found.append(item)
        if found:
            return found

        # Méthode 3 : fallback → retourner toute la liste
        return fallback_list

    # ============================================================
    #  LLM POUR CHOISIR LES TABLES À EXTRAIRE
    # ============================================================

    def select_tables_with_llm(self, prompt_custom=None):
        """
        Utilise le LLM pour choisir les tables à extraire avec prompt personnalisé
        """
        # 1. Récupérer la liste des tables
        tables_df = self.get_all_tables()
        table_list = tables_df['TABLE_NAME'].tolist()

        if not table_list:
            raise ValueError("Aucune table trouvée dans la base de données.")

        # 2. Prompt système par défaut
        system_prompt = """
You are a data architect and database expert.
Your task is to analyze database table names and select the most relevant ones for extraction.
Consider factors like:
- Tables that contain core business data
- Tables likely to be used in data analysis
- Tables that might need cleaning or quality checks
- Avoid system tables, logs, or temporary tables

Return ONLY a valid Python list of table names, nothing else.
Example: ["customers", "orders", "products"]
"""

        # 3. Prompt utilisateur (personnalisable)
        if prompt_custom:
            user_prompt = f"""
Database tables: {table_list}

Additional requirements: {prompt_custom}

Which tables should be extracted? Return only a Python list.
"""
        else:
            user_prompt = f"""
Database tables: {table_list}

Which tables are most important for data analysis and should be extracted?
Return only a Python list.
"""

        # 4. Appeler le LLM
        raw_response = self._call_llm(system_prompt, user_prompt)

        # 5. Parser la réponse
        selected_tables = self._parse_llm_list(raw_response, table_list)

        return {
            "tables_in_db": table_list,
            "selected_tables": selected_tables,
            "llm_raw_response": raw_response,
            "prompt_used": user_prompt
        }

    def extract_with_llm_selection(self, file_path, prompt_custom=None):
        """
        Extrait les tables sélectionnées par le LLM
        """
        # Obtenir la sélection du LLM
        selection = self.select_tables_with_llm(prompt_custom)
        
        # Extraire les tables choisies
        os.makedirs(file_path, exist_ok=True)
        extracted = []

        for table in selection["selected_tables"]:
            if table in selection["tables_in_db"]:
                df = self.extract_table(table)
                df.to_csv(f"{file_path}/{table}.csv", index=False)
                extracted.append(table)
            else:
                print(f"[WARN] Table '{table}' suggérée mais introuvable")

        if not extracted:
            # Fallback
            print("[WARN] Aucune table extraite, extraction de toutes les tables")
            for table in selection["tables_in_db"]:
                df = self.extract_table(table)
                df.to_csv(f"{file_path}/{table}.csv", index=False)
                extracted.append(table)

        return {
            "tables_in_db": selection["tables_in_db"],
            "selected_tables": selection["selected_tables"],
            "extracted_tables": extracted,
            "llm_raw_response": selection["llm_raw_response"],
            "prompt_used": selection["prompt_used"]
        }

    # ============================================================
    #  LLM POUR FILTRER LES DONNÉES AVEC PROMPT
    # ============================================================

    def generate_filter_conditions(self, table_name, prompt_custom=None):
        """
        Utilise le LLM pour générer des conditions de filtrage sur une table
        """
        # 1. Récupérer le schéma de la table
        schema_df = self.extract_table_schema(table_name)
        
        if schema_df.empty:
            raise ValueError(f"Impossible de récupérer le schéma de {table_name}")

        # Construire la description du schéma
        schema_desc = []
        for _, row in schema_df.iterrows():
            schema_desc.append(f"- {row['COLUMN_NAME']} ({row['DATA_TYPE']})")
        
        schema_text = "\n".join(schema_desc)

        # 2. Échantillon de données (premières lignes)
        try:
            sample_df = self.extract_data(f"SELECT * FROM {table_name} LIMIT 5")
            sample_text = sample_df.to_string()
        except:
            sample_text = "Impossible d'échantillonner les données"

        # 3. Prompt système
        system_prompt = """
You are a data analyst and SQL expert.
Your task is to generate relevant filter conditions for a database table.
The filters should help extract meaningful subsets of data for analysis.

Return ONLY a valid JSON object with filter conditions, nothing else.
Format: 
{
    "filters": [
        {
            "column": "column_name",
            "operator": "=" | "!=" | ">" | "<" | ">=" | "<=" | "LIKE" | "IN" | "BETWEEN",
            "value": value or [values],
            "description": "why this filter is useful"
        }
    ],
    "logic": "AND" | "OR",
    "explanation": "brief explanation of the filtering strategy"
}
"""

        # 4. Prompt utilisateur
        if prompt_custom:
            user_prompt = f"""
Table: {table_name}

Schema:
{schema_text}

Sample data (first 5 rows):
{sample_text}

User request: {prompt_custom}

Generate appropriate filter conditions for this table.
"""
        else:
            user_prompt = f"""
Table: {table_name}

Schema:
{schema_text}

Sample data (first 5 rows):
{sample_text}

Generate useful filter conditions to extract relevant data for analysis.
Consider:
- Removing outliers
- Focusing on recent data
- Selecting specific categories
- Filtering out null values

Return only the JSON object with filters.
"""

        # 5. Appeler le LLM
        raw_response = self._call_llm(system_prompt, user_prompt, temperature=0.2)

        # 6. Parser la réponse JSON
        try:
            # Essayer d'extraire le JSON
            match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if match:
                filters = json.loads(match.group())
            else:
                filters = json.loads(raw_response)
        except Exception as e:
            # Fallback : créer un filtre simple
            filters = {
                "filters": [],
                "logic": "AND",
                "explanation": "Échec du parsing LLM, aucun filtre généré",
                "error": str(e)
            }

        return {
            "table": table_name,
            "schema": schema_desc,
            "filters": filters,
            "llm_raw_response": raw_response,
            "prompt_used": user_prompt
        }

    def extract_with_llm_filters(self, table_name, file_path, prompt_custom=None):
        """
        Extrait une table avec des filtres générés par LLM
        """
        # 1. Obtenir les filtres du LLM
        filter_result = self.generate_filter_conditions(table_name, prompt_custom)
        
        # 2. Construire la requête SQL avec les filtres
        filters = filter_result["filters"].get("filters", [])
        logic = filter_result["filters"].get("logic", "AND")
        
        if filters:
            where_clauses = []
            for f in filters:
                column = f.get("column")
                operator = f.get("operator")
                value = f.get("value")
                
                if not column or not operator:
                    continue
                
                # Formater la valeur selon le type
                if operator.upper() == "LIKE":
                    where_clauses.append(f"{column} LIKE '{value}'")
                elif operator.upper() == "IN":
                    if isinstance(value, list):
                        values_str = ", ".join([f"'{v}'" if isinstance(v, str) else str(v) for v in value])
                        where_clauses.append(f"{column} IN ({values_str})")
                elif operator.upper() == "BETWEEN":
                    if isinstance(value, list) and len(value) == 2:
                        where_clauses.append(f"{column} BETWEEN {value[0]} AND {value[1]}")
                else:
                    # Opérateurs standards
                    if isinstance(value, str) and not value.isdigit():
                        where_clauses.append(f"{column} {operator} '{value}'")
                    else:
                        where_clauses.append(f"{column} {operator} {value}")
            
            if where_clauses:
                where_sql = f" WHERE {' ' + logic + ' '.join(where_clauses)}"
                query = f"SELECT * FROM {table_name}{where_sql}"
            else:
                query = f"SELECT * FROM {table_name}"
        else:
            query = f"SELECT * FROM {table_name}"

        # 3. Exécuter la requête
        try:
            df = self.extract_data(query)
            path = f"{file_path}/{table_name}_filtered.csv"
            df.to_csv(path, index=False)
            
            return {
                "success": True,
                "table": table_name,
                "query": query,
                "rows_extracted": len(df),
                "filter_conditions": filters,
                "saved_to": path,
                "filter_result": filter_result
            }
        except Exception as e:
            return {
                "success": False,
                "table": table_name,
                "error": str(e),
                "query": query,
                "filter_result": filter_result
            }

    # ============================================================
    #  LLM POUR ANALYSE COMPLÈTE AVEC PROMPT
    # ============================================================

    def analyze_with_llm(self, prompt):
        """
        Analyse complète : le LLM décide quoi extraire ET comment filtrer
        """
        # 1. Récupérer toutes les tables
        tables_df = self.get_all_tables()
        all_tables = tables_df['TABLE_NAME'].tolist()

        # 2. Récupérer les schémas de toutes les tables
        schemas = {}
        for table in all_tables[:5]:  # Limiter à 5 tables pour éviter de surcharger
            try:
                schema_df = self.extract_table_schema(table)
                schemas[table] = [
                    f"{row['COLUMN_NAME']} ({row['DATA_TYPE']})" 
                    for _, row in schema_df.iterrows()
                ]
            except:
                schemas[table] = ["Schema indisponible"]

        # 3. Construire le prompt
        schema_text = ""
        for table, cols in schemas.items():
            schema_text += f"\n{table}:\n"
            for col in cols[:10]:  # Limiter à 10 colonnes par table
                schema_text += f"  - {col}\n"

        system_prompt = """
You are a senior data architect and AI analyst.
Your task is to analyze a database and decide:
1. Which tables to extract
2. For each selected table, what filters to apply
3. The purpose of this extraction

Return a valid JSON object with your analysis.
"""

        user_prompt = f"""
Database tables and schemas:
{schema_text}

User request: {prompt}

Based on this request, provide:
- tables_to_extract: list of table names
- for each table, filters: array of filter conditions
- overall_purpose: brief explanation

Return ONLY a JSON object.
"""

        # 4. Appeler le LLM
        raw_response = self._call_llm(system_prompt, user_prompt, temperature=0.1)

        # 5. Parser la réponse
        try:
            match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if match:
                analysis = json.loads(match.group())
            else:
                analysis = json.loads(raw_response)
        except:
            analysis = {
                "tables_to_extract": all_tables[:3],
                "filters": {},
                "overall_purpose": "Analyse par défaut",
                "raw_response": raw_response
            }

        return {
            "analysis": analysis,
            "all_tables": all_tables,
            "prompt": prompt,
            "llm_raw_response": raw_response
        }