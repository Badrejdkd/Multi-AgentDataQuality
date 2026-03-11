import logging
import os
import sys
import os
import pandas as pd
from sqlalchemy import text
from langchain_ollama import ChatOllama
from langchain.messages import SystemMessage, HumanMessage
 
# ajouter la racine du projet au path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agents.extraction_agent import ExtractionAgent
from agents.quality_agent import QualityAgent
from agents.cleaning_agent import CleaningAgent
from agents.storage_agent import StorageAgent
from llm.llm_agent import LLMAgent



# configuration du logging
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class Orchestrator:

    def __init__(self, db_type, db_name):

        logging.info("Initializing Multi-Agent System")

        self.extractor = ExtractionAgent(db_type, db_name)
        self.quality = QualityAgent()
        self.cleaner = CleaningAgent()
        self.storage = StorageAgent()
        self.llm = LLMAgent()


    # pipeline complet
    def run_pipeline(self):

        logging.info("Starting pipeline")

        tables_df = self.extractor.get_all_tables()

        table_list = tables_df["TABLE_NAME"].tolist()

        logging.info(f"Tables detected: {table_list}")

        # LLM décide quelles tables nettoyer
        tables_to_clean = self.llm.select_tables_for_cleaning(table_list)

        logging.info(f"Tables selected for cleaning: {tables_to_clean}")

        for table in table_list:

            logging.info(f"Processing table: {table}")

            df = self.extractor.extract_table(table)

            # sauvegarder raw
            self.storage.save_raw_table(df, table)

            # analyse qualité
            report = self.quality.quality_report(df)

            self.storage.save_quality_report(report, table)

            logging.info(f"Quality report for {table}: {report}")

            # si la table doit être nettoyée
            if table in tables_to_clean:

                logging.info(f"Cleaning table: {table}")

                # LLM suggère une stratégie
                strategy = self.llm.suggest_cleaning_strategy(report)

                logging.info(f"Suggested cleaning strategy: {strategy}")

                cleaned_df = self.cleaner.clean_table(df)

                self.storage.save_cleaned_table(cleaned_df, table)

        logging.info("Pipeline finished successfully")


    # analyser seulement la qualité
    def run_quality_analysis(self):

        tables_df = self.extractor.get_all_tables()

        reports = {}

        for table in tables_df["TABLE_NAME"]:

            df = self.extractor.extract_table(table)

            report = self.quality.quality_report(df)

            reports[table] = report

        return reports


    # traiter une seule table
    def run_single_table(self, table_name):

        logging.info(f"Processing single table: {table_name}")

        df = self.extractor.extract_table(table_name)

        report = self.quality.quality_report(df)

        cleaned_df = self.cleaner.clean_table(df)

        self.storage.save_cleaned_table(cleaned_df, table_name)

        return cleaned_df