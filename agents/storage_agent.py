import os


class StorageAgent:

    def save_to_csv(self, df, file_path):

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        df.to_csv(file_path, index=False)

        return file_path


    def save_raw_table(self, df, table_name):

        path = f"data/raw/{table_name}.csv"

        return self.save_to_csv(df, path)


    def save_cleaned_table(self, df, table_name):

        path = f"data/cleaned/{table_name}.csv"

        return self.save_to_csv(df, path)


    def save_quality_report(self, report, table_name):

        os.makedirs("reports", exist_ok=True)

        path = f"reports/{table_name}_quality.txt"

        with open(path, "w") as f:

            f.write(str(report))

        return path