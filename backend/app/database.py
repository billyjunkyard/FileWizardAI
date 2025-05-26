import sqlite3
import hashlib
import json # For storing lists as JSON strings


class SQLiteDB:
    def __init__(self):
        self.conn = sqlite3.connect('FileWizardAi.db')
        # Enable foreign key enforcement if desired, though not strictly used in schema
        # self.cursor.execute("PRAGMA foreign_keys = ON;")
        self.cursor = self.conn.cursor()
        
        # Updated files_summary table
        files_summary_table_query = """
        CREATE TABLE IF NOT EXISTS files_summary (
            file_path TEXT PRIMARY KEY,
            file_hash TEXT NOT NULL,
            summary TEXT,
            research_topic TEXT,
            is_topic_relevant BOOLEAN,
            sub_topics TEXT,
            topic_connections TEXT,
            analysis_type TEXT,
            analysis_llm_prompt TEXT,
            file_analysis_hash TEXT 
        )"""
        self.cursor.execute(files_summary_table_query)

        # New document_chunks table
        document_chunks_table_query = """
        CREATE TABLE IF NOT EXISTS document_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL,
            chunk_order INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            vector_embedding BLOB
        )"""
        self.cursor.execute(document_chunks_table_query)

        # Index for document_chunks on file_path
        create_index_query = "CREATE INDEX IF NOT EXISTS idx_chunk_file_path ON document_chunks (file_path);"
        self.cursor.execute(create_index_query)
        
        self.conn.commit()

    def select(self, table_name, where_clause=None):
        sql = f"SELECT * FROM {table_name}"
        if where_clause:
            sql += f" WHERE {where_clause}"
        self.cursor.execute(sql)
        return self.cursor.fetchall()

    def is_file_exist(self, file_path, file_hash):
        self.cursor.execute("SELECT * FROM files_summary WHERE file_path = ? AND file_hash = ?", (file_path, file_hash))
        file = self.cursor.fetchone()
        return bool(file)

    def insert_file_summary(self, file_path, file_hash, summary,
                            research_topic=None, is_topic_relevant=None,
                            sub_topics=None, topic_connections=None,
                            analysis_type=None, analysis_llm_prompt=None,
                            file_analysis_hash=None):
        c = self.conn.cursor()
        c.execute("SELECT file_path FROM files_summary WHERE file_path=?", (file_path,))
        existing_file = c.fetchone()

        # Convert sub_topics list to JSON string if it's a list
        if isinstance(sub_topics, list):
            sub_topics_json = json.dumps(sub_topics)
        else:
            sub_topics_json = sub_topics # Assume it's already a JSON string or None

        if existing_file:
            # Update existing record
            update_fields = {
                "file_hash": file_hash,
                "summary": summary,
                "research_topic": research_topic,
                "is_topic_relevant": is_topic_relevant,
                "sub_topics": sub_topics_json,
                "topic_connections": topic_connections,
                "analysis_type": analysis_type,
                "analysis_llm_prompt": analysis_llm_prompt,
                "file_analysis_hash": file_analysis_hash,
            }
            # Filter out fields that are None, so they don't overwrite existing values with NULL
            # unless explicitly provided as None.
            # However, the task implies we want to update these fields even if they are None (to clear them).
            # So, we'll include all of them in the update.
            
            set_clause = ", ".join([f"{key}=?" for key in update_fields.keys()])
            values = list(update_fields.values())
            values.append(file_path)
            
            c.execute(f"UPDATE files_summary SET {set_clause} WHERE file_path=?", values)
        else:
            # Insert new record
            c.execute("""
                INSERT INTO files_summary (
                    file_path, file_hash, summary, research_topic, is_topic_relevant,
                    sub_topics, topic_connections, analysis_type, analysis_llm_prompt,
                    file_analysis_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (file_path, file_hash, summary, research_topic, is_topic_relevant,
                  sub_topics_json, topic_connections, analysis_type, analysis_llm_prompt,
                  file_analysis_hash))
        self.conn.commit()

    def get_file_summary(self, file_path):
        self.cursor.execute("SELECT summary FROM files_summary WHERE file_path = ?", (file_path,))
        result = self.cursor.fetchone()
        return result[0] if result else None

    def drop_table(self): # Consider dropping document_chunks as well if used for full reset
        self.cursor.execute("DROP TABLE IF EXISTS files_summary")
        self.cursor.execute("DROP TABLE IF EXISTS document_chunks") # Added
        self.conn.commit()

    def get_all_files(self):
        self.cursor.execute("SELECT file_path FROM files_summary")
        results = self.cursor.fetchall()
        files_path = [row[0] for row in results]
        return files_path

    def update_file(self, old_file_path, new_file_path, new_hash):
        # This should also update file_path in document_chunks if it's a primary concern.
        # For now, assuming file renames might require re-chunking/re-processing.
        # If strict FKs were used with ON UPDATE CASCADE, this would be automatic.
        self.cursor.execute("UPDATE files_summary SET file_path = ?, file_hash = ? WHERE file_path = ?",
                            (new_file_path, new_hash, old_file_path))
        # Update associated chunks:
        self.cursor.execute("UPDATE document_chunks SET file_path = ? WHERE file_path = ?",
                            (new_file_path, old_file_path))
        self.conn.commit()

    def delete_records(self, file_paths):
        if not file_paths:
            return
        placeholders = ",".join("?" * len(file_paths))
        self.cursor.execute(f"DELETE FROM files_summary WHERE file_path IN ({placeholders})", file_paths)
        self.cursor.execute(f"DELETE FROM document_chunks WHERE file_path IN ({placeholders})", file_paths) # Added
        self.conn.commit()

    # New methods for document chunks and analysis data
    def insert_document_chunks(self, file_path: str, chunks: list[dict]):
        # Delete existing chunks for this file to prevent duplicates
        self.cursor.execute("DELETE FROM document_chunks WHERE file_path = ?", (file_path,))
        
        chunk_data_to_insert = []
        for chunk_dict in chunks:
            chunk_data_to_insert.append((
                file_path,
                chunk_dict.get('chunk_order'),
                chunk_dict.get('chunk_text')
                # vector_embedding is initially NULL
            ))
        
        if chunk_data_to_insert:
            self.cursor.executemany(
                "INSERT INTO document_chunks (file_path, chunk_order, chunk_text) VALUES (?, ?, ?)",
                chunk_data_to_insert
            )
            self.conn.commit()

    def get_document_chunks(self, file_path: str) -> list[dict]:
        self.cursor.execute(
            "SELECT id, file_path, chunk_order, chunk_text, vector_embedding FROM document_chunks WHERE file_path = ? ORDER BY chunk_order ASC",
            (file_path,)
        )
        rows = self.cursor.fetchall()
        column_names = [description[0] for description in self.cursor.description]
        
        results = []
        for row in rows:
            row_dict = dict(zip(column_names, row))
            if row_dict.get('vector_embedding') is not None:
                try:
                    # Deserialize from JSON string bytes to list[float]
                    embedding_json_bytes = row_dict['vector_embedding']
                    embedding_json_str = embedding_json_bytes.decode('utf-8')
                    row_dict['vector_embedding'] = json.loads(embedding_json_str)
                except (json.JSONDecodeError, UnicodeDecodeError, AttributeError) as e:
                    # Log error or handle as per application's error policy
                    # For now, set to None if deserialization fails to prevent crash
                    print(f"Error deserializing embedding for chunk ID {row_dict.get('id')}: {e}") # Basic print for now
                    row_dict['vector_embedding'] = None 
            results.append(row_dict)
        return results

    def update_chunk_embedding(self, chunk_id: int, embedding: bytes):
        # The 'embedding' parameter is expected to be bytes, 
        # e.g., sqlite3.Binary(json.dumps(embedding_vector).encode('utf-8'))
        self.cursor.execute("UPDATE document_chunks SET vector_embedding = ? WHERE id = ?", (embedding, chunk_id))
        self.conn.commit()

    def get_file_analysis_data(self, file_path: str, expected_file_analysis_hash: str) -> dict | None:
        """
        Retrieves file analysis data if the provided expected_file_analysis_hash matches the stored hash.
        """
        self.cursor.execute("""
            SELECT research_topic, is_topic_relevant, sub_topics, topic_connections, analysis_type, file_analysis_hash
            FROM files_summary 
            WHERE file_path = ? AND file_analysis_hash = ?
        """, (file_path, expected_file_analysis_hash))
        
        row = self.cursor.fetchone()
        if row:
            column_names = [description[0] for description in self.cursor.description]
            analysis_data = dict(zip(column_names, row))
            
            # Deserialize sub_topics from JSON string to list
            if analysis_data.get('sub_topics'):
                try:
                    analysis_data['sub_topics'] = json.loads(analysis_data['sub_topics'])
                except json.JSONDecodeError:
                    # Handle cases where sub_topics might not be valid JSON (e.g. old data or error)
                    analysis_data['sub_topics'] = [] # Or log an error
            return analysis_data
        return None

    def close(self):
        self.conn.close()
