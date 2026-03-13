class CleaningAgent:

    def remove_duplicates(self, df):
        before = len(df)
        df = df.drop_duplicates()
        # ── retourner aussi le nombre de doublons supprimés ──
        return df, before - len(df)

    def fill_missing_values(self, df):
        filled = 0
        for col in df.columns:
            missing = int(df[col].isnull().sum())
            if missing > 0:
                filled += missing
                if df[col].dtype == "object":
                    df[col] = df[col].fillna("unknown")
                else:
                    df[col] = df[col].fillna(df[col].median())
        # ── retourner aussi le nombre de valeurs remplies ──
        return df, filled

    def normalize_text(self, df):
        text_cols = list(df.select_dtypes(include="object").columns)
        for col in text_cols:
            df[col] = df[col].str.strip().str.lower()
        # ── retourner aussi les colonnes normalisées ──
        return df, text_cols

    def clean_table(self, df):
        df, dups   = self.remove_duplicates(df)
        df, filled = self.fill_missing_values(df)
        df, cols   = self.normalize_text(df)
        # ── retourner df + stats ──
        return df, {
            "duplicates_removed":  dups,
            "missing_filled":      filled,
            "columns_normalized":  cols,
        }