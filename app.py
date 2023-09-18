"""
This module contains the Flask application for your project.

It handles routes, templates, and database connections.
"""
import os
import secrets
import re
import io
import openai
import spacy
import torch
import PyPDF2
from PyPDF2 import PdfFileReader, PdfFileWriter,PageObject
from bs4 import BeautifulSoup 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from flask import redirect, url_for, Flask, render_template, request, jsonify, session, send_file
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.pdfmetrics import stringWidth
from docx import Document
from werkzeug.utils import secure_filename
from transformers import BertTokenizer, BertForSequenceClassification
from db_connection import DBConnection
import numpy as np
from joblib import load
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

model = load('random_forest.joblib')
vectorizer = load('tfidf_vectorizer.joblib')


# Mapping of section indices to section names
section_names = {0: "Introduction", 1: "Method", 2: "Result", 3: "Discussion"}

# Initializing the Flask application
app = Flask(__name__)

# Generating a secure secret key for the session
app.secret_key = secrets.token_hex(16)

# Loading the spaCy NLP model for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")

# Setting up the database connection
DB = 'database.db'

# Loading the BERT model and tokenizer from a specified path
model_path = "Models\BERT Model" 
tokenizer = BertTokenizer.from_pretrained(model_path)
bert_model = BertForSequenceClassification.from_pretrained(model_path)

# Setting up OpenAI API credentials
openai.api_key = "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"

# Configuring upload settings for the Flask application
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'doc', 'docx'}

# Specifying the full path to database.db
db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database.db')

# Creating an instance of DBConnection for interacting with the database
db_connection = DBConnection(db_path)


@app.route('/')
def index():
    """Route for the home page"""
    # Rendering the home page using a template named 'index.html'
    return render_template('index.html')

@app.route('/validate_login', methods=['POST'])
def validate_login():
    """Route for validating login credentials"""
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
    """Route for retrieving the session state"""
    # Checking if the user is logged in by looking for 'email' in the session
    isLoggedIn = 'email' in session
    
    # Returning the login state as JSON
    return jsonify({'isLoggedIn': isLoggedIn})

@app.route('/logout', methods=['POST'])
def logout():
    """Route for logging out the user"""
    # Clearing all data from the session to log out the user
    session.clear()
    
    # Returning a success message as JSON
    return jsonify({'success': True})

@app.route('/forget-password')
def forget_password():
    """
    Render the forget password page.

    This route displays the forget password page where users can request a password reset.
    """
    # Rendering the forget password page using a template named 'fp.html'
    return render_template('fp.html')

@app.route('/browse', methods=['GET'])
def browse():
    """
    Browse publications.

    This route handles the browsing of publications based on selected sorting, field, year, and search query.
    """
    try:
        # Creating a DBConnection instance for interacting with the database
        db_conn = DBConnection('database.db')

        # Extracting query parameters from the request
        selected_sort = request.args.get('sort', 'latest')
        selected_field = request.args.get('field', '')
        selected_year = request.args.get('year', '')
        search_query = request.args.get('search', '')  # Get the search query from the query parameters

        # Fetching publications from the database based on query parameters
        items, unique_subject_areas, unique_years = db_conn.fetch_publications(
            selected_sort, selected_field, selected_year, search_query)

        # Closing the database connection
        db_conn.close_connection()

        # Rendering the browse page with fetched publications and query parameters
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
    """Route for rendering the upload form"""
    # Rendering the fromdocx page using a template named 'fromdocx.html'
    return render_template("fromdocx.html")

@app.route("/publish")
def publish():
    """Route for rendering the upload form"""
    # Rendering the publish page using a template named 'publish.html'
    return render_template("publish.html")

@app.route("/chatbot")
def chatbot():
    """Route for rendering the chatbot form"""
    # Rendering the chatbot page using a template named 'chatbot.html'
    return render_template("chatbot.html")

@app.route('/DV')
def DV():
    """Route for the documentation page"""
    # Rendering the documentation page using a template named 'DV.html'
    return render_template('DV.html')

@app.route("/research")
def research():
    """Route for rendering the unapproved docx"""
    try:
        # Creating a DBConnection instance for interacting with the database
        db_conn = DBConnection('database.db')

        # Extracting query parameters from the request
        search_query = request.args.get('search', '')  # Get the search query from the query parameters

        # Fetching research publications from the database based on query parameters
        items = db_conn.fetch_research_publications(search_query)

        # Closing the database connection
        db_conn.close_connection()

        # Rendering the research page with fetched publications and query parameters
        return render_template('research.html', items=items, search_query=search_query)
    except Exception as e:
        print("Error browsing publications:", e)
        
        # Rendering an error page if an exception occurs
        return render_template('404.html')


@app.route("/api", methods=["POST"])
def api():
    """Route to handle POST requests for the chatbot API"""
    # Extracting message from the POST request
    message = request.json.get("message")
    
    # Sending the message to OpenAI's API and receiving the response
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": message}
        ]
    )
    
    # Returning the response message if it exists, otherwise returning an error message
    if completion.choices[0].message is not None:
        return completion.choices[0].message
    else:
        return 'Failed to Generate response!'

def save_file(file):
    """
    Function to handle file uploads.

    This function saves the uploaded file to the specified UPLOAD_FOLDER.
    """
    if file:
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return filename
    return None

@app.route('/submit_data', methods=['POST'])
def submit_data():
    """
    Route to handle form submission.

    This route handles form submission from 'publish.html'.
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

    This route handles form upload from 'fromdocx.html'.
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
    
    if record is not None:
        # Extracting data from the record
        title = record['title']
        # ... extract other data from record ...
        
        # Rendering the abstract page with extracted data
        return render_template('abstract.html', title=title)  # Pass data to template
    else:
        # Returning an error message if no unapproved records are found
        return "No unapproved records found."
    
@app.route('/get_last_unapproved')
def get_last_unapproved_route():
    record = db_connection.get_last_unapproved()
    return jsonify(record)

# Define the custom order
section_order = ['Introduction', 'Method', 'Result', 'Discussion']

# Defining a route for the generate_abstract function
@app.route('/generate_abstract')
def generate_abstract():
    threshold = 0.6
    print("Starting to generate abstract...")
    record = db_connection.get_last_unapproved()

    if record is not None:
        print("Record found, processing...")
        file_path = record['file_path'] 
        doc = Document(file_path)
        doc_text = " ".join([p.text for p in doc.paragraphs])
        soup = BeautifulSoup(doc_text, 'html.parser')
        cleaned_text = re.sub(r'\s+', ' ', soup.get_text(separator=' '))
        chunk_size = 512
        text_chunks = [cleaned_text[i:i+chunk_size] for i in range(0, len(cleaned_text), chunk_size)]
        section_texts = {section: "" for section in section_names.values()}
        section_texts['Other'] = ""
        batch_size = 20

        for i in range(0, len(text_chunks), batch_size):
            print(f"Processing batch {i//batch_size + 1}...")
            batch = text_chunks[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)

            with torch.no_grad():
                outputs = bert_model(**inputs)
                predicted_sections = torch.argmax(outputs.logits, dim=1)

            # Initialize a dictionary to store the maximum probabilities for each section
            max_probs = {section: 0 for section in section_order}
            max_probs['Other'] = 0

            # After getting the predictions
            for j, input_ids in enumerate(inputs["input_ids"]):
                # Get the maximum probability and its corresponding section
                max_prob, predicted_section = torch.max(outputs.logits[j]), predicted_sections[j].item()
                current_section = section_names[predicted_section]

                # Check if the maximum probability is below the threshold
                if max_prob.item() < threshold:
                    current_section = 'Other'

                # Decode the entire batch of tokens at once
                token_text = tokenizer.decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

                # Only add to the section if it's not already present or if the current probability is higher than the previous maximum
                if current_section not in section_texts[current_section] or max_prob.item() > max_probs[current_section]:
                    # Remove the section name and a colon before the decoded text
                    section_texts[current_section] = f"\n\n{token_text}"
                    max_probs[current_section] = max_prob.item()

                section_texts[current_section] = section_texts[current_section].replace(' ##', '')

            # Create a list of text strings with the sections in your desired order
            sorted_section_texts = [section_texts[section] for section in section_order if section in section_texts]
            
            print("Finished processing. Sending response...")
            return jsonify(sorted_section_texts)




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
    words = text.split(' ')
    
    lines = []
    
    currentLine = ''
    
    for word in words:
        if stringWidth(currentLine + ' ' + word, fontName, fontSize) <= maxWidth:
            currentLine += ' ' + word
        else:
            lines.append(currentLine)
            currentLine = word
            
    if currentLine:
        lines.append(currentLine)
        
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