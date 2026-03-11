import pandas as pd


class QualityAgent:

    def detect_missing_values(self, df):

        missing = df.isnull().sum()

        return missing.to_dict()


    def detect_duplicates(self, df):

        duplicates = df.duplicated().sum()

        return duplicates


    def detect_outliers(self, df):

        outliers = {}

        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

        for col in numeric_cols:

            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)

            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            outliers[col] = ((df[col] < lower) | (df[col] > upper)).sum()

        return outliers


    def quality_score(self, df):

        total_cells = df.shape[0] * df.shape[1]

        missing = df.isnull().sum().sum()

        duplicates = df.duplicated().sum()

        score = 100 - ((missing + duplicates) / total_cells * 100)

        return round(score, 2)


    def quality_report(self, df):

        report = {
            "rows": len(df),
            "columns": len(df.columns),
            "missing_values": self.detect_missing_values(df),
            "duplicates": self.detect_duplicates(df),
            "outliers": self.detect_outliers(df),
            "quality_score": self.quality_score(df)
        }

        return report