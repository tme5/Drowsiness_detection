import sqlite3
import time

class SqliteDB():
    def __init__(self):
        self.conn = sqlite3.connect('detection.db')
        self.cursor = self.conn.cursor()
        self.cursor.execute("DROP TABLE IF EXISTS detection_data")
                                
    def insert_data(self, counter, EAR, YAWN):
        with self.conn:
            self.cursor.execute("CREATE TABLE IF NOT EXISTS detection_data (counter integer, EAR real, YAWN real)")
            self.cursor.execute("DELETE FROM detection_data WHERE counter<:limit", {'limit': counter-900})
            self.cursor.execute("INSERT INTO detection_data VALUES (:counter, :EAR, :YAWN)", {'counter': counter, 'EAR': EAR, 'YAWN': YAWN})
    
    def get_all_data(self):
        counter = time.time()- 900
        self.cursor.execute("SELECT * FROM detection_data WHERE counter>=:counter", {'counter': counter})
        return self.cursor.fetchall()
    
    def get_yawn_th(self, YAWN_TH):
        self.cursor.execute("SELECT * FROM detection_data WHERE YAWN=:YAWN_TH", {'YAWN_TH': YAWN_TH})
        return self.cursor.fetchall()

    def get_ear_th(self, EAR_TH):
        self.cursor.execute("SELECT * FROM detection_data WHERE EAR=:EAR_TH", {'EAR_TH': EAR})
        return self.cursor.fetchall()

    def conn_close(self):
        self.conn.close()