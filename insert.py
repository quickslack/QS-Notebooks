import pandas as pd
from tqdm import tqdm
import psycopg2
from sqlalchemy import create_engine

db_string = "postgresql://postgres:postgres@postgres/postgres"
db = create_engine(db_string)

df = pd.read_csv('cleaned.csv')

conn = db.raw_connection()
cur = conn.cursor()

for _, row in tqdm(df.iterrows()):
    cur.execute('INSERT INTO cleaned VALUES(%s, %s)', (row['message_id'], row['cleaned']))
conn.commit()
conn.close()
