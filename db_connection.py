import sqlite3
from flask import g

class DBConnection:
    def __init__(self, DB):
        self.DB = DB

    def get_db(self):
        db = getattr(g, '_database', None)
        if db is None:
            db = g._database = sqlite3.connect(self.DB)
        return db
    
    def execute_query(self, query, params=None):
        conn = self.get_db()
        cursor = conn.cursor()
        if params is None:
            cursor.execute(query)
        else:
            cursor.execute(query, params)
        conn.commit()
        return cursor
    
    def fetch_all(self, cursor):
        return cursor.fetchall()

    def insert_into_working(self, title, file_path):
        try:
            query = "INSERT INTO working (title, File_Path) VALUES (?, ?)"
            self.execute_query(query, (title, file_path))
            return True  # Return True if insertion was successful
        except Exception as e:
            print(f"An error occurred: {e}")
            return False  # Return False if an error occurred


    def close_connection(self):
        db = getattr(g, '_database', None)
        if db is not None:
            db.close()

    def validate_login(self, email, password):
        """
        Validate login credentials by querying the database.

        Parameters:
        - email (str): The email to validate.
        - password (str): The password to validate.

        Returns:
        - email_exists (bool): True if the email exists in the database, False otherwise.
        - password_match (bool): True if the password matches the email in the database, False otherwise.
        """
        # Query the database to validate the login credentials
        conn = self.get_db()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM users WHERE email=?", (email,))
        result = cursor.fetchone()

        if result:
            stored_password = result[4]  # Assuming the password is stored in the fourth column
            if password == stored_password:
                return True, True
            else:
                return True, False
        else:
            return False, False

    def email_exists(self, email):
        """
        Check if an email already exists in the database.

        Parameters:
        - email (str): The email to check.

        Returns:
        - exists (bool): True if the email exists in the database, False otherwise.
        """
        conn = self.get_db()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM users WHERE email=?", (email,))
        return cursor.fetchone()[0] > 0

    def create_account(self, lname, fname, email, password):
        """
        Create a new account in the database.

        Parameters:
        - lastName (str): The last name of the user.
        - firstName (str): The first name of the user.
        - email (str): The email of the user.
        - password (str): The password of the user.

        Returns:
        - success (bool): True if the account is created successfully, False otherwise.
        - message (str): A message indicating the result of the account creation.
        """
        # Check if the email already exists
        if self.email_exists(email):
            return False, 'Email already exists.'

        # Create the account
        conn = self.get_db()
        cursor = conn.cursor()
        
        cursor.execute("INSERT INTO users (lname, fname, email, password) VALUES (?, ?, ?, ?)",
                            (lname, fname, email, password))
        conn.commit()

        return True, 'Account created successfully.'
    
    def insert_upload(self, title, authors, publicationDate, thesisAdvisor, department, degree, subjectArea, abstract, file_path):
        """
        Insert a new record into the 'uploads' table.

        Parameters:
        - title (str): The title of the research.
        - authors (str): The authors of the research.
        - publicationDate (str): The publication date of the research.
        - thesisAdvisor (str): The thesis advisor's name.
        - department (str): The department related to the research.
        - degree (str): The degree associated with the research.
        - subjectArea (str): The subject area of the research.
        - abstract (str): The abstract of the research.
        - file_path (str): The path of the uploaded file in the server.

        Returns:
        - bool: True if the insertion is successful, False otherwise.
        """
        try:
            conn = self.get_db()
            cursor = conn.cursor()

            cursor.execute("INSERT INTO uploads (title, authors, publicationDate, thesisAdvisor, department, degree, subjectArea, abstract, file_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", 
                        (title, authors, publicationDate, thesisAdvisor, department, degree, subjectArea, abstract, file_path))
            conn.commit()
            return True
        except Exception as e:
            print("Error inserting record:", e)
            conn.rollback()
            return False


        
    def fetch_publications(self, selected_sort, selected_field, selected_year, search_query):
        try:
            conn = self.get_db()
            cursor = conn.cursor()

            query = "SELECT id, title, authors, publicationDate, subjectArea FROM uploads"

            # Filter based on the selected field option
            params = ()
            if selected_field:
                query += " AND subjectArea = ?"
                params += (selected_field,)

            cursor.execute(query, params)
            items = [{'id': row[0], 'title': row[1], 'authors': row[2], 'year': row[3], 'subjectArea': row[4]} for row in cursor.fetchall()]

            # Fetch unique subject areas from the database
            cursor.execute("SELECT DISTINCT subjectArea FROM uploads")
            unique_subject_areas = [row[0] for row in cursor.fetchall()]

            # Fetch unique years from the database
            cursor.execute("SELECT DISTINCT substr(publicationDate, 1, 4) FROM uploads")
            unique_years = [row[0] for row in cursor.fetchall()]

            # Filter items by year if selected_year is provided
            if selected_year:
                items = [item for item in items if item['year'] == selected_year]

            # Sort the items based on the selected_sort option
            if selected_sort == 'latest':
                items.sort(key=lambda x: x['year'], reverse=True)
            elif selected_sort == 'oldest':
                items.sort(key=lambda x: x['year'])
            else:  # Sort alphabetically by title (default)
                items.sort(key=lambda x: x['title'])

            # Filter items by search query
            if search_query:
                search_query = search_query.lower()
                items = [item for item in items if search_query in item['title'].lower() or search_query in item['authors'].lower()]

            return items, unique_subject_areas, unique_years
        except Exception as e:
            print("Error fetching publications:", e)
            return [], [], []

    def fetch_research_publications(self, search_query):
        try:
            conn = self.get_db()
            cursor = conn.cursor()

            # Modify this query to match the actual columns in your table
            query = "SELECT id, title FROM working"

            cursor.execute(query)
            items = [{'id': row[0], 'title': row[1]} for row in cursor.fetchall()]

            # Filter items by search query
            if search_query:
                search_query = search_query.lower()
                items = [item for item in items if search_query in item['title'].lower()]

            return items
        except Exception as e:
            print("Error fetching publications:", e)
            return []




        
    def get_publication_by_id(self, item_id):
        try:
            conn = self.get_db()
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM uploads WHERE id=?", (item_id,))
            row = cursor.fetchone()

            if row:
                item = {
                    'id': row[0],
                    'title': row[1],
                    'authors': row[2],
                    'year': row[3],
                    'thesisAdvisor': row[4],
                    'department': row[5],
                    'degree': row[6],
                    'subjectArea': row[7],
                    'abstract': row[8],
                    'file_path': row[9]
                }
                return item
            else:
                return None
        except Exception as e:
            print("Error fetching publication:", e)
            return None
        
    def update_converted_file_path(self, item_id, converted_file_path):
        try:
            conn = self.get_db()
            cursor = conn.cursor()

            # Update the 'converted_file_path' column in the 'uploads' table
            cursor.execute("UPDATE uploads SET converted_file_path=? WHERE id=?", (converted_file_path, item_id))
            conn.commit()
            return True
        except Exception as e:
            print("Error updating converted file path:", e)
            conn.rollback()
            return False

    def get_last_unapproved(self):
        try:
            conn = self.get_db()
            conn.row_factory = sqlite3.Row  # Use sqlite3.Row to access columns by name
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM working ORDER BY id DESC LIMIT 1")
            record = cursor.fetchone()
            return dict(record) if record else None  # Convert the Row to a dict
        except Exception as e:
            print("Error retrieving record:", e)
            return None
        
    def update_abstract(self, id, abstract):
        query = 'UPDATE working SET abstract = ? WHERE id = ?'
        params = (abstract, id)
        self.execute_query(query, params)

