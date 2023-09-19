# This module contains the Flask application for your project.
# It handles routes, templates, and database connections.

# Importing necessary libraries
import os  # Provides functions for interacting with the operating system
import secrets  # Used for generating cryptographically strong random numbers
import re  # Provides regular expression matching operations
import io  # Provides the Python interfaces to stream handling
import openai  # OpenAI's API for accessing GPT-3 and other models
import spacy  # Library for advanced Natural Language Processing in Python
import torch  # PyTorch is a Python package that provides two high-level features: tensor computation with strong GPU acceleration and deep neural networks built on a tape-based autograd system
import PyPDF2  # A Pure-Python library built as a PDF toolkit
import random  # This module implements pseudo-random number generators for various distributions
from PyPDF2 import PdfFileReader, PdfFileWriter,PageObject  # Classes for reading and writing PDF files
from bs4 import BeautifulSoup  # Library for pulling data out of HTML and XML files 
from nltk.corpus import stopwords  # NLTK corpus stopwords
from nltk.tokenize import word_tokenize, sent_tokenize  # Tokenizers divide strings into lists of substrings
from flask import redirect, url_for, Flask, render_template, request, jsonify, session, send_file  # Flask web server framework
from reportlab.pdfgen import canvas  # Library to create PDF documents using Python
from reportlab.lib.pagesizes import A4  # Standard paper sizes like A4
from reportlab.pdfbase.pdfmetrics import stringWidth  # Function to calculate string width in PDF metrics
from docx import Document  # Create or modify MS Word files using Python 
from werkzeug.utils import secure_filename  # Werkzeug utility wrappers, including secure file naming
from transformers import BertTokenizer, BertForSequenceClassification  # Transformers provides state-of-the-art general-purpose architectures (BERT) for Natural Language Understanding (NLU) and Natural Language Generation (NLG)
from db_connection import DBConnection  # Custom module to handle database connections
import numpy as np  # NumPy is a Python library used for working with arrays. It also has functions for working in domain of linear algebra, fourier transform, and matrices.
import torch.nn.functional as F  # PyTorch's functional interface 
from joblib import load  # Joblib is a set of tools to provide lightweight pipelining in Python. It provides utilities for saving and loading Python objects that make use of NumPy data structures.
from collections import Counter  # A Counter is a dict subclass for counting hashable objects.
from sklearn.ensemble import RandomForestClassifier  # The RandomForestClassifier from sklearn.ensemble module 
from sklearn.feature_extraction.text import TfidfVectorizer  # Convert a collection of raw documents to a matrix of TF-IDF features with TfidfVectorizer

# Initializing the Flask application 
app = Flask(__name__)

# Generating a secure secret key for the session 
app.secret_key = secrets.token_hex(16)

# Configuring upload settings for the Flask application 
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'doc', 'docx'}

# Setting up OpenAI API credentials 
openai.api_key = "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"

# Loading the trained Random Forest model and TF-IDF vectorizer from disk 
model = load('random_forest.joblib')
vectorizer = load('tfidf_vectorizer.joblib')

# Loading the spaCy NLP model for Named Entity Recognition (NER) 
nlp = spacy.load("en_core_web_sm")

# Loading the BERT model and tokenizer from a specified path 
model_path = "Models\BERT Model" 
tokenizer = BertTokenizer.from_pretrained(model_path)
bert_model = BertForSequenceClassification.from_pretrained(model_path)

# Mapping of section indices to section names for document sections 
section_names = {0: "Introduction", 1: "Method", 2: "Result", 3: "Discussion"}

# Define the custom order of sections in the document 
section_order = ['Introduction', 'Method', 'Result', 'Discussion']

# Setting up the database connection 
DB = 'database.db'

# Specifying the full path to database.db 
db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database.db')

# Creating an instance of DBConnection for interacting with the database 
db_connection = DBConnection(db_path)


@app.route('/')
def index():
    """
    Route for the home page.
    This route renders the home page using a template named 'index.html'.
    """
    return render_template('index.html')


@app.route('/validate_login', methods=['POST'])
def validate_login():
    """
    Route for validating login credentials.
    This route extracts login data from the request, validates the credentials using a DBConnection instance,
    and returns a response indicating the success of login validation and the login state.
    """
    
    # Creating a DBConnection instance for interacting with the database
    db = DBConnection(db_path)

    # Extracting login data from the request
    data = request.json
    email = data['email']
    password = data['password']

    # Validating login credentials using the DBConnection instance
    email_exists, password_match = db.validate_login(email, password)
    
    # Closing the database connection
    db.close_connection()

    # If both email exists and password matches, store user information in the session
    if email_exists and password_match:
        session['email'] = email
        isLoggedIn = True
    else:
        isLoggedIn = False

    # Preparing a response indicating the success of login validation and the login state
    response = {
        'success': email_exists and password_match,
        'emailExists': email_exists,
        'passwordMatch': password_match,
        'isLoggedIn': isLoggedIn
    }

    # Returning the response as JSON
    return jsonify(response)


@app.route('/session_state')
def session_state():
    """
    Route for retrieving the session state.
    This route checks if the user is logged in by looking for 'email' in the session and returns the login state as JSON.
    """
    
    # Checking if the user is logged in by looking for 'email' in the session
    isLoggedIn = 'email' in session
    
    # Returning the login state as JSON
    return jsonify({'isLoggedIn': isLoggedIn})


@app.route('/logout', methods=['POST'])
def logout():
    """
    Route for logging out the user.
    This route clears all data from the session to log out the user and returns a success message as JSON.
    """
    
    # Clearing all data from the session to log out the user
    session.clear()
    
    # Returning a success message as JSON
    return jsonify({'success': True})


@app.route('/forget-password')
def forget_password():
    """
    Route for the forget password page.
    This route displays the forget password page where users can request a password reset.

    Rendering: The forget password page using a template named 'fp.html'
    Return: Rendered template 
    """

    return render_template('fp.html')


@app.route('/browse', methods=['GET'])
def browse():
    """
    Route for browsing publications.
    This route handles browsing of publications based on selected sorting, field, year, and search query.

    Try-Except block: To handle any exceptions that might occur during execution of code within it. 
    """

    try:
        # Creating a DBConnection instance for interacting with the database
        db_conn = DBConnection('database.db')

        # Extracting query parameters from the request
        selected_sort = request.args.get('sort', 'latest')
        selected_field = request.args.get('field', '')
        selected_year = request.args.get('year', '')
        search_query = request.args.get('search', '')  # Get search query from query parameters

        # Fetching publications from database based on query parameters
        items, unique_subject_areas, unique_years = db_conn.fetch_publications(
            selected_sort, selected_field, selected_year, search_query)

        # Closing database connection
        db_conn.close_connection()

        # Rendering browse page with fetched publications and query parameters
        return render_template('browse.html', items=items, subject_areas=unique_subject_areas,
                                unique_years=unique_years, selected_sort=selected_sort,
                                selected_field=selected_field, selected_year=selected_year,
                                search_query=search_query)
    except Exception as e:
        print("Error browsing publications:", e)
        
        # Rendering an error page if an exception occurs
        return render_template('404.html')


@app.route("/fromdocx")
def fromdocx():
    """
    Route for rendering the upload form.
    This route displays a form where users can upload a document.
    
    Rendering: The fromdocx page using a template named 'fromdocx.html'
    Return: Rendered template 
    """
    
    return render_template("fromdocx.html")


@app.route("/publish")
def publish():
    """
    Route for rendering the upload form.
    This route displays a form where users can upload a publication.
    
    Rendering: The publish page using a template named 'publish.html'
    Return: Rendered template 
    """
    
    return render_template("publish.html")


@app.route("/chatbot")
def chatbot():
    """
    Route for rendering the chatbot form.
    This route displays a form where users can interact with a chatbot.
    
    Rendering: The chatbot page using a template named 'chatbot.html'
    Return: Rendered template 
    """
    
    return render_template("chatbot.html")


@app.route('/DV')
def DV():
    """
    Route for the documentation page.
    This route displays a documentation page for users to learn more about the application.

    Rendering: The documentation page using a template named 'DV.html'
    Return: Rendered template 
    """

    return render_template('DV.html')


@app.route("/research")
def research():
    """
    Route for rendering the unapproved docx.
    This route fetches research publications from the database based on a search query and renders a research page with the fetched publications.

    Try-Except block: To handle any exceptions that might occur during execution of code within it. 
    """

    try:
        # Creating a DBConnection instance for interacting with the database
        db_conn = DBConnection('database.db')

        # Extracting search query from the request
        search_query = request.args.get('search', '')  

        # Fetching research publications from the database based on search query
        items = db_conn.fetch_research_publications(search_query)

        # Closing the database connection
        db_conn.close_connection()

        # Rendering the research page with fetched publications and search query
        return render_template('research.html', items=items, search_query=search_query)
    except Exception as e:
        print("Error browsing publications:", e)
        
        # Rendering an error page if an exception occurs
        return render_template('404.html')


@app.route("/api", methods=["POST"])
def api():
    """
    Route to handle POST requests for the chatbot API.
    This route extracts a message from the POST request, sends it to OpenAI's API, and returns the response message.

    Extracting: Message from POST request
    Sending: Message to OpenAI's API and receiving response
    Returning: Response message if it exists, otherwise returning an error message
    """

    message = request.json.get("message")  # Extracting message from POST request

    completion = openai.ChatCompletion.create(  # Sending message to OpenAI's API and receiving response
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": message}
        ]
    )

    if completion.choices[0].message is not None:
        return completion.choices[0].message  # Returning response message if it exists
    else:
        return 'Failed to Generate response!'  # Returning error message


def save_file(file):
    """
    Function to handle file uploads.
    This function saves uploaded file to specified UPLOAD_FOLDER and returns its filename.

    If file exists:
        Saving: Uploaded file to specified UPLOAD_FOLDER
        Returning: Filename of uploaded file
    Else:
        Returning: None
    """

    if file:
        filename = file.filename  # Filename of uploaded file
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))  # Saving uploaded file to specified UPLOAD_FOLDER
        return filename  # Returning filename of uploaded file
    return None  # If no file exists, returning None


@app.route('/submit_data', methods=['POST'])
def submit_data():
    """
    Route to handle form submission.
    This route handles form submission from 'publish.html'. It extracts data from the form, saves any uploaded file,
    inserts a new upload record into the database, and returns a success message.
    
    Extracting: Data from the form
    Saving: Uploaded file and getting its filename
    Inserting: New upload record into the database
    Returning: Success message
    """
    
    title = request.form['title']
    authors = request.form['authors']
    publicationDate = request.form['publicationDate']
    thesisAdvisor = request.form['thesisAdvisor']
    department = request.form['department']
    degree = request.form['degree']
    subjectArea = request.form['subjectArea']
    abstract = request.form['abstract']
    uploaded_file = request.files['file']

    # Saving uploaded file and getting its filename
    filename = save_file(uploaded_file)

    if filename:
        # Inserting upload record into database and checking if insertion was successful
        if db_connection.insert_upload(title, authors, publicationDate, thesisAdvisor, department, degree, subjectArea, abstract, filename):
            print("Upload record inserted successfully!")
            return "Upload successful!"
        else:
            print("Failed to insert upload record.")

    return "Data submitted successfully!"


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Route to handle form upload.
    This route handles form upload from 'fromdocx.html'. It extracts the title and file from the form data,
    validates them, saves the uploaded file, inserts a new upload record into the database, and returns a success message.
    """
    try:
        # Get the title from the form data
        title = request.form.get('title')
        
        # Get the file from the form data
        file = request.files.get('file')

        # Check if both title and file are provided
        if not title or not file:
            return 'Please fill in all fields.', 400

        # Check if the uploaded file is a .docx file
        if not file.filename.endswith('.docx'):
            return 'Invalid file type. Please upload a .docx file.', 400

        # Secure the filename and get the path to save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Save the uploaded file to the specified path
        file.save(file_path)

        # Create a DBConnection instance and insert the title and file path into the "working" table
        db_conn = DBConnection('database.db')
        
        if filename:
            # Inserting upload record into database and checking if insertion was successful
            if db_conn.insert_into_working(title, file_path):
                print("Upload record inserted successfully!")
                return "Upload successful!"
            else:
                print("Failed to insert upload record.")
                return "Upload failed.", 500

        db_conn.close_connection()

    except Exception as e:
        print(f"An error occurred: {e}")
        return "An error occurred while processing your request.", 500

    return "Data submitted successfully!"


@app.route('/abstract')
def abstract():
    """
    Route for rendering the abstract of the last unapproved record.
    This route fetches the last unapproved record from the database and renders an abstract page with its data.
    If no unapproved records are found, it returns an error message.
    """
    
    # Fetching the last unapproved record from the database
    record = db_connection.get_last_unapproved()
    
    # Checking if a record was found
    if record is not None:
        
        # Extracting data from the record
        title = record['title']
        
        # Rendering the abstract page with extracted data
        return render_template('abstract.html', title=title)  # Pass data to template
    
    else:
        # Returning an error message if no unapproved records are found
        return "No unapproved records found."
    
    
@app.route('/get_last_unapproved')
def get_last_unapproved_route():
    """
    Route for fetching the last unapproved record.
    This route fetches the last unapproved record from the database and returns it as a JSON object.
    """
    
    # Fetching the last unapproved record from the database
    record = db_connection.get_last_unapproved()
    
    # Returning the fetched record as a JSON object
    return jsonify(record)


@app.route('/upload_abstract', methods=['POST'])
def upload_abstract_route():
    """
    Route for uploading an abstract.
    This route receives a JSON object containing an abstract, fetches the last unapproved record from the database,
    and updates its abstract with the received one. It returns a JSON object indicating whether the operation was successful.
    """
    
    # Extracting the abstract from the received JSON object
    data = request.get_json()
    abstract = data['abstract']
    
    # Fetching the last unapproved record from the database
    record = db_connection.get_last_unapproved()
    
    # Checking if a record was found
    if record:
        # Updating the record with the new abstract
        success = db_connection.update_abstract(record['id'], abstract)
        
        # Checking if the update operation was successful and returning a corresponding status message
        if success:
            return jsonify({'status': 'success'})
    
    return jsonify({'status': 'failure'})


@app.route('/generate_abstract')
def generate_abstract():
    """
    Route for generating an abstract.
    This route fetches the last unapproved record from the database, divides its text into chunks, classifies each chunk,
    selects the top chunks for each section, and generates an abstract based on these chunks.
    """
    threshold = 0.4
    print("Starting to generate abstract...")
    
    # Fetching the last unapproved record from the database
    record = db_connection.get_last_unapproved()

    if record is not None:
        print("Record found, processing...")
        
        # Extracting file path from the record
        file_path = record['file_path'] 
        
        # Reading text from .docx file
        doc = Document(file_path)
        doc_text = " ".join([p.text for p in doc.paragraphs])
        
        # Cleaning up text
        soup = BeautifulSoup(doc_text, 'html.parser')
        cleaned_text = re.sub(r'\s+', ' ', soup.get_text(separator=' '))
        
        # Splitting text into sentences
        sentences = sent_tokenize(cleaned_text)
        
        # Dividing sentences into chunks of up to 512 tokens each
        chunk_size = 512
        text_chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += " " + sentence
            else:
                text_chunks.append(current_chunk)
                current_chunk = sentence

        if current_chunk:
            text_chunks.append(current_chunk)

        section_texts = {section: [] for section in section_names.values()}
        section_texts['Other'] = []
        
        batch_size = 20

        # Classify all chunks first
        for i in range(0, len(text_chunks), batch_size):
            print(f"Classifying batch {i//batch_size + 1}...")
            
            batch = text_chunks[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)

            with torch.no_grad():
                outputs = bert_model(**inputs)
                predicted_sections = torch.argmax(outputs.logits, dim=1)

            for j, input_ids in enumerate(inputs["input_ids"]):
                token_text = tokenizer.decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                current_section = section_names[predicted_sections[j].item()]
                section_texts[current_section].append((token_text, outputs.logits[j]))

        # Process each section
        total_probability = 0
        total_chunks = 0
        sorted_section_texts = {}

        # Process each section
        for section in section_order:
            if section in section_texts:
                print(f"Processing section {section}...")
                
                # Apply softmax and get top 5 chunks
                top_chunks = sorted(section_texts[section], key=lambda x: F.softmax(x[1], dim=0).max().item(), reverse=True)[:5]
                
                # Randomly select a chunk from the top 5
                selected_chunk, logits = random.choice(top_chunks)
                
                sorted_section_texts[section] = f"\n\n{selected_chunk}"

                # Add the max softmax probability of the selected chunk to the total
                total_probability += F.softmax(logits, dim=0).max().item()
                total_chunks += 1

        # Calculate the average probability
        average_probability = total_probability / total_chunks if total_chunks > 0 else 0
        
        print(f"Average probability of chosen chunks: {average_probability}")


@app.route('/publication_detail/<int:item_id>')
def publication_detail(item_id):
    """
    Route for displaying publication details.
    This route fetches the publication details from the database based on the item_id and displays them.
    """
    
    # Creating a DBConnection instance for interacting with the database
    db_conn = DBConnection(DB)
    
    # Fetching the publication details from the database based on the item_id
    item = db_conn.get_publication_by_id(item_id)
    
    # Closing the database connection
    db_conn.close_connection()

    if item is not None:
        # Constructing the path to the PDF file of the publication
        pdf_path = os.path.join('uploads', item['file_path'])
        
        # Reading the text content of the PDF file using PyPDF2
        text_content = read_pdf_text(pdf_path)

        # Rendering the publication detail page with fetched publication details and text content of the PDF file
        return render_template('publication_detail.html', item=item, text_content=text_content)
    
    else:
        # Rendering an error page if the publication with the specified ID is not found
        return render_template('404.html'), 404


def read_pdf_text(pdf_path):
    """
    Function to read the text content of a PDF file using PyPDF2.
    This function reads the text content of a PDF file and returns it as a string.
    """
    text_content = ""
    
    # Opening the PDF file in binary read mode
    with open(pdf_path, 'rb') as file:
        
        # Creating a PdfReader instance for reading the PDF file
        pdf_reader = PyPDF2.PdfReader(file)
        
        # Getting the number of pages in the PDF file
        num_pages = len(pdf_reader.pages)
        
        # Reading text content from each page of the PDF file and appending it to text_content string
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text_content += page.extract_text()

    return text_content


@app.route('/uploads/<path:filename>')
def serve_pdf(filename):
    """
    Route to serve PDF files.
    This route serves the specified PDF file from the 'uploads' directory.
    """
    
    # Constructing the path to the PDF file
    pdf_path = os.path.join('uploads', filename)
    
    # Sending the PDF file as response with 'application/pdf' MIME type
    return send_file(pdf_path, mimetype='application/pdf')

def generate_apa_citation_from_data(publication):
    """
    Function to generate APA citation from publication data.
    This function takes the publication data and returns the APA citation in the specified format.
    """
    
    # Splitting authors' names and getting the number of authors
    authors = publication['authors'].split('; ')
    num_authors = len(authors)

    # Formatting authors' names based on the number of authors
    if num_authors == 1:
        formatted_authors = authors[0].split()[-1]
    elif num_authors == 2:
        formatted_authors = f"{authors[0].split()[-1]} & {authors[1].split()[-1]}"
    else:
        formatted_authors = ", ".join(author.split()[-1] for author in authors[:-1])
        formatted_authors += f", & {authors[-1].split()[-1]}"

    # Formatting the APA citation based on the publication data
    apa_citation = f"{formatted_authors}. ({publication['year']}). {publication['title']}. {publication['thesisAdvisor']}. {publication['department']}. {publication['degree']}."

    return apa_citation


@app.route('/generate_apa_citation/<int:item_id>')
def generate_apa_citation(item_id):
    """
    Route to generate APA citation for a publication.
    This route takes the item_id of a publication, generates the APA citation, and returns it as a JSON response.
    """
    
    # Creating a DBConnection instance for interacting with the database
    db_conn = DBConnection(DB)
    
    # Fetching the publication by its ID from the database
    publication = db_conn.get_publication_by_id(item_id)

    if publication:
        # Generating the APA citation based on the publication data
        apa_citation = generate_apa_citation_from_data(publication)

        # Returning the APA citation as a JSON response
        return jsonify({"apa_citation": apa_citation})

    return jsonify({"error": "Publication not found"})


def simpleSplit(text, fontName, fontSize, maxWidth):
    """
    Function to split text into lines based on font size and maximum width.
    This function splits text into lines such that each line's width does not exceed the specified maximum width.
    """
    
    # Split the input text into words
    words = text.split(' ')
    
    # Initialize an empty list to hold the lines of text
    lines = []
    
    # Initialize an empty string to hold the current line of text
    currentLine = ''
    
    # Iterate over each word in the list of words
    for word in words:
        # If adding a new word to the current line doesn't exceed the maximum width
        if stringWidth(currentLine + ' ' + word, fontName, fontSize) <= maxWidth:
            # Add the word to the current line
            currentLine += ' ' + word
        else:
            # If adding a new word to the current line exceeds the maximum width,
            # add the current line to the list of lines and start a new line with the current word
            lines.append(currentLine)
            currentLine = word
            
    # If there's any text left in the current line after iterating through all the words, add it as a new line
    if currentLine:
        lines.append(currentLine)
        
    # Return the list of lines
    return lines


def clean_text(text):
    """
     Function to clean text.
     This function removes non-alphanumeric characters, converts to lowercase, and removes stopwords from text.
     """
    
    # Removing non-alphanumeric characters
    text = re.sub(r'\W+', ' ', text)
     
    # Converting to lowercase
    text = text.lower()
    
    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    
    word_tokens = word_tokenize(text)
    
    text = [word for word in word_tokens if not word in stop_words]
    
    return ' '.join(text)


def convert_to_imrad(file_path):
    """
    Function to convert a document to IMRaD format.
    This function reads a PDF file, extracts its text content, cleans the text, splits it into chunks, processes the chunks using a BERT model to predict sections, organizes the chunks into different sections based on predictions, cleans the section texts, and writes the section texts into a new PDF file in IMRaD format.
    """
    try:
        # Opening the PDF file in binary read mode
        with open(file_path, "rb") as f:
            # Creating a PdfReader instance for reading the PDF file
            pdf_reader = PdfFileReader(f)
            
            # Reading text content from each page of the PDF file and appending it to pdf_text string
            pdf_text = ""
            for page in pdf_reader.pages:
                pdf_text += page.extract_text()

        # Cleaning the extracted text
        pdf_text = clean_text(pdf_text)

        chunk_size = 512
        
        # Splitting the cleaned text into chunks of 512 characters each for processing by BERT model
        text_chunks = [pdf_text[i:i+chunk_size] for i in range(0, len(pdf_text), chunk_size)]
        
        # Initializing a dictionary to store section texts
        section_texts = {section: "" for section in section_names.values()}

        # Defining batch size for processing by BERT model
        batch_size = 10

        # Processing text chunks in batches
        for i in range(0, len(text_chunks), batch_size):
            # Preparing inputs for BERT model by tokenizing text chunks in current batch
            batch = text_chunks[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            
            with torch.no_grad():
                # Running BERT model on inputs and getting outputs without gradient computation (for efficiency)
                outputs = bert_model(**inputs)
                
                # Predicting sections for each input by taking argmax of logits from outputs
                predicted_sections = torch.argmax(outputs.logits, dim=1)
                
            for j, input_ids in enumerate(inputs["input_ids"]):
                current_section = section_names[predicted_sections[j].item()]
                for token in input_ids:
                    token_text = tokenizer.decode([token.item()], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    
                    if current_section is not None:
                        if section_texts[current_section] and not section_texts[current_section].endswith(' '):
                            section_texts[current_section] += ' '
                        
                        section_texts[current_section] += token_text

                print(f"Section: {current_section}")
                print(f"Text: {section_texts[current_section]}")
                print(f"Token: {token_text}")
                
        # Cleaning the extracted text
        section_texts[current_section] = clean_text(section_texts[current_section])            
        
        # Removing unnecessary spaces added by BERT tokenizer (represented as ' ##' in tokenized text)
        section_texts[current_section] = section_texts[current_section].replace(' ##', '')                               
        
        converted_file_path = file_path.replace(os.path.splitext(file_path)[1], '_imrad.pdf')
        
        pdf_writer = PdfFileWriter()

        for section_name, section_text in section_texts.items():
            # Create a new page
            page = PageObject.createBlankPage(None, 595.44, 841.68)  # A4 size in points
            
            # Add text to the page
            packet = io.BytesIO()
            can = canvas.Canvas(packet, pagesize=A4)
            
            textobject = can.beginText()
            
            # Set a 1-inch margin
            textobject.setTextOrigin(72, 841.68 - 72)  # A4 size in points, 1 inch = 72 points
            
            textobject.setFont("Helvetica", 10)
            
            # Add section name
            textobject.textLine(section_name)
            
            # Add section text
            for line in section_text.split('\n'):
                # Wrap text if it's too long for one line
                lines = simpleSplit(line, "Helvetica", 10, 595.44 - 144)  # Subtract margins from page width
                
                for l in lines:
                    textobject.textLine(l)
            
            can.drawText(textobject)
            
            can.save()

            # Move to the beginning of the StringIO buffer
            packet.seek(0)
            
            new_pdf = PdfFileReader(packet)
            
            # Add the "watermark" (which is the new pdf) on the existing page
            page.mergePage(new_pdf.getPage(0))
            
            # Add page to the writer
            pdf_writer.addPage(page)

        # Write output file
        with open(converted_file_path, "wb") as out_f:
            pdf_writer.write(out_f)

        print(f"Converted file saved at {converted_file_path}")
        
        return converted_file_path
    except Exception as e:
        print(f"An error occurred: {str(e)}")


@app.route('/convert_to_imrad/<int:item_id>', methods=['GET'])
def convert_to_imrad_route(item_id):
    """
    Route to convert a publication to IMRAD format.

    This route takes the item_id of a publication, converts it to IMRAD format, and returns the converted PDF.
    """
    # Fetching the publication by its ID from the database
    publication = db_connection.get_publication_by_id(item_id)
    
    if publication is None:
        # Returning an error message as JSON response if the publication with the specified ID is not found
        return jsonify({"message": "Publication not found."}), 404

    # Constructing the path to the PDF file of the publication
    file_name = os.path.basename(publication['file_path'])
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    
    try:
        # Converting the PDF file to IMRaD format and getting the path to the converted file
        converted_file_path = convert_to_imrad(file_path)

        # Updating the database with the path to the converted file
        db_connection.update_converted_file_path(item_id, converted_file_path)

        # Returning the converted PDF file as response without attachment (for preview in page)
        return send_file(converted_file_path, as_attachment=False)
    except Exception as e:
        # Returning an error message as JSON response if an exception occurs during conversion
        return jsonify({"message": "Error converting to IMRAD.", "error": str(e)}), 500


if __name__ == '__main__':
    app.run()