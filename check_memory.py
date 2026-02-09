import sqlite3
import pandas as pd

# Path to your database
DB_PATH = "backend/db/memory_store.sqlite"

def read_memory():
    try:
        # Connect to database
        conn = sqlite3.connect(DB_PATH)
        
        # Check tables
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Tables found: {tables}")

        # The table is named 'memory' in your current service
        query = "SELECT * FROM memory" 
        
        df = pd.read_sql_query(query, conn)
        
        if df.empty:
            print("\nMemory is empty. Did you ask a question yet?")
        else:
            print(f"\n--- Chat History ({len(df)} turns) ---")
            
            # Display the relevant columns for your schema
            # Schema: id, conv_id, ts, role, content, meta
            if 'conv_id' in df.columns:
                print(df[['conv_id', 'role', 'content']].tail(10))
            else:
                # Fallback if using the old schema
                print(df.tail(5))
            
        conn.close()

    except Exception as e:
        print(f"Error reading database: {e}")

if __name__ == "__main__":
    read_memory()