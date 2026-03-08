from sqlalchemy import create_engine

def connect_mysql(db):

    engine = create_engine(
        f"mysql+pymysql://root:password@localhost:3306/{db}"
    )

    return engine

def connect_postgresql(db):

    engine = create_engine(
        f"postgresql+psycopg2://postgres:password@localhost/{db}"
    )

    return engine

def connect_sqlite(db):

    engine = create_engine(
        f"sqlite:///{db}.sqlite"
    )

    return engine

def connect_oracle(db):

    engine = create_engine(
        f"oracle+cx_oracle://username:password@localhost:1521/{db}"
    )

    return engine

from sqlalchemy import create_engine

def connect_sql_server(db):

    server = "DESKTOP-MEOKMIT\\SQLEXPRESS"
    driver = "ODBC Driver 17 for SQL Server"

    connection_string = (
        f"mssql+pyodbc://@{server}/{db}"
        f"?driver={driver.replace(' ', '+')}"
        "&trusted_connection=yes"
    )

    engine = create_engine(connection_string)

    return engine