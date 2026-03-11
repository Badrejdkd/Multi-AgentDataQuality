from langchain_ollama import ChatOllama
from langchain.messages import SystemMessage, HumanMessage


class LLMAgent:

    def __init__(self, model="llama2", temperature=0):

        self.model = ChatOllama(
            model=model,
            temperature=temperature
        )


    # décider quelles tables doivent être nettoyées
    def select_tables_for_cleaning(self, table_list):

        messages = [
            SystemMessage(
                content="""
You are a data quality expert.

Your job is to analyze database table names and decide which tables
are likely to require cleaning (missing values, duplicates, bad formatting).

Return ONLY a Python list of table names.
"""
            ),
            HumanMessage(
                content=f"Database tables: {table_list}"
            )
        ]

        response = self.model.invoke(messages)

        try:
            tables = eval(response.content)
        except:
            tables = []

        return tables


    # suggérer une stratégie de nettoyage
    def suggest_cleaning_strategy(self, quality_report):

        messages = [
            SystemMessage(
                content="""
You are a senior data engineer.
Suggest a data cleaning strategy based on the quality report.
"""
            ),
            HumanMessage(
                content=f"Quality report: {quality_report}"
            )
        ]

        response = self.model.invoke(messages)

        return response.content


    # analyser la structure d'une table
    def analyze_table_schema(self, schema):

        messages = [
            SystemMessage(
                content="""
You are a database expert.
Analyze the schema and detect potential data quality issues.
"""
            ),
            HumanMessage(
                content=f"Table schema: {schema}"
            )
        ]

        response = self.model.invoke(messages)

        return response.content


    # générer un résumé des données
    def summarize_dataset(self, profile):

        messages = [
            SystemMessage(
                content="""
You are a data analyst.
Summarize the dataset profile in simple terms.
"""
            ),
            HumanMessage(
                content=f"Dataset profile: {profile}"
            )
        ]

        response = self.model.invoke(messages)

        return response.content