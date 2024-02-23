import sqlite3

engine = sqlite3.connect('configs/text_app.db', check_same_thread=False)
cur = engine.cursor()

def create_table():
    cur.execute("""
        CREATE TABLE IF NOT EXISTS input_text(
            id INTEGER PRIMARY KEY,
            text TEXT NOT NULL,
            app NVARCHAR(20) NOT NULL
        )
    """)
    

def add_one_item(text, app, table='input_text'):
    cur.execute(f"INSERT INTO {table} (text, app) VALUES (?, ?)", (text, app,))
    engine.commit()


def get_all_data(table='input_text'):
    data = cur.execute('SELECT * FROM {}'.format(table))
    return data.fetchall()

