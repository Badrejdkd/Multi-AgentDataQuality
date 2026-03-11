class CleaningAgent:

    def remove_duplicates(self, df):

        df = df.drop_duplicates()

        return df


    def fill_missing_values(self, df):

        for col in df.columns:

            if df[col].dtype == "object":

                df[col] = df[col].fillna("unknown")

            else:

                df[col] = df[col].fillna(df[col].median())

        return df


    def normalize_text(self, df):

        text_columns = df.select_dtypes(include="object").columns

        for col in text_columns:

            df[col] = df[col].str.strip().str.lower()

        return df


    def clean_table(self, df):

        df = self.remove_duplicates(df)

        df = self.fill_missing_values(df)

        df = self.normalize_text(df)

        return df