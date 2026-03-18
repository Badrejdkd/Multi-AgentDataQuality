# agents/cleaning_agent.py - VERSION CORRIGÉE

import pandas as pd
import numpy as np

class CleaningAgent:

    def remove_duplicates(self, df):
        """Supprime les doublons et retourne le DataFrame avec les stats"""
        before = len(df)
        df = df.drop_duplicates()
        return df, before - len(df)

    def fill_missing_values(self, df):
        """Remplit les valeurs manquantes de façon adaptée au type de données"""
        filled = 0
        
        for col in df.columns:
            missing = int(df[col].isnull().sum())
            if missing > 0:
                filled += missing
                
                # Pour les colonnes textuelles/objets
                if df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col]):
                    # Remplacer par la valeur la plus fréquente (mode) ou "unknown"
                    if not df[col].mode().empty:
                        most_frequent = df[col].mode()[0]
                        df[col] = df[col].fillna(most_frequent)
                    else:
                        df[col] = df[col].fillna("unknown")
                
                # Pour les colonnes numériques
                elif pd.api.types.is_numeric_dtype(df[col]):
                    # Remplacer par la médiane (ignorer les NaN)
                    median_val = df[col].median(skipna=True)
                    if pd.isna(median_val):  # Si toutes les valeurs sont NaN
                        median_val = 0
                    df[col] = df[col].fillna(median_val)
                
                # Pour les colonnes booléennes
                elif pd.api.types.is_bool_dtype(df[col]):
                    df[col] = df[col].fillna(False)
                
                # Pour les colonnes datetime
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = df[col].fillna(pd.NaT)
                
                # Autres types
                else:
                    df[col] = df[col].fillna("unknown")
        
        return df, filled

    def normalize_text(self, df):
        """Normalise le texte : strip, lower, et gestion des valeurs nulles"""
        text_cols = []
        
        for col in df.select_dtypes(include=["object"]).columns:
            # Vérifier que la colonne contient du texte
            if df[col].dtype == "object":
                text_cols.append(col)
                # Appliquer strip et lower uniquement sur les chaînes non-nulles
                df[col] = df[col].apply(lambda x: x.strip().lower() if isinstance(x, str) else x)
        
        return df, text_cols

    def clean_table(self, df):
        """Nettoie la table complète avec toutes les opérations"""
        # Sauvegarder les stats
        stats = {}
        
        # 1. Supprimer les doublons
        df, stats['duplicates_removed'] = self.remove_duplicates(df)
        
        # 2. Remplir les valeurs manquantes
        df, stats['missing_filled'] = self.fill_missing_values(df)
        
        # 3. Normaliser le texte
        df, stats['columns_normalized'] = self.normalize_text(df)
        
        return df, stats