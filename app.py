"""
This module contains the Flask application for your project.

It handles routes, templates, and database connections.
"""
import os
import re
import secrets
import io
import openai
import spacy
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from docx import Document
from docx.shared import Pt
import PyPDF2
from fpdf import FPDF
from flask import Flask, render_template, request, jsonify, session, send_file
from werkzeug.utils import secure_filename
from db_connection import DBConnection

app = Flask(__name__)

# Generate a secure secret key for the session
app.secret_key = secrets.token_hex(16)
# Load the spaCy NLP model for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")
DB = 'database.db'

# Set up OpenAI API credentials
openai.api_key = "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"

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'doc', 'docx'}
# Specify the full path to database.db
db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database.db')
db_connection = DBConnection(db_path)  # Create an instance of DBConnection


@app.route('/')
def index():
    """Route for the home page"""
    return render_template('index.html')


@app.route('/validate_login', methods=['POST'])
def validate_login():
    """Route for validating login credentials"""
    db = DBConnection(db_path)  # Create DBConnection instance

    data = request.json
    email = data['email']
    password = data['password']

    # Perform login validation using the DBConnection instance
    email_exists, password_match = db.validate_login(email, password)
    db.close_connection()

    if email_exists and password_match:
        # Store user information in the session
        session['email'] = email
        isLoggedIn = True
    else:
        isLoggedIn = False

    response = {
        'success': email_exists and password_match,
        'emailExists': email_exists,
        'passwordMatch': password_match,
        'isLoggedIn': isLoggedIn
    }

    return jsonify(response)


@app.route('/session_state')
def session_state():
    """Route for retrieving the session state"""
    isLoggedIn = 'email' in session
    return jsonify({'isLoggedIn': isLoggedIn})


@app.route('/logout', methods=['POST'])
def logout():
    """Route for logging out the user"""
    session.clear()  # Clear the session data
    return jsonify({'success': True})


@app.route('/forget-password')
def forget_password():
    """
    Render the forget password page.

    This route displays the forget password page where users can request a password reset.
    """
    return render_template('fp.html')


@app.route('/browse', methods=['GET'])
def browse():
    """
    Browse publications.

    This route handles the browsing of publications based on selected sorting, field, year, and search query.
    """
    try:
        db_conn = DBConnection('database.db')

        selected_sort = request.args.get('sort', 'latest')
        selected_field = request.args.get('field', '')
        selected_year = request.args.get('year', '')
        search_query = request.args.get('search', '')  # Get the search query from the query parameters

        items, unique_subject_areas, unique_years = db_conn.fetch_publications(
            selected_sort, selected_field, selected_year, search_query)

        db_conn.close_connection()

        return render_template('browse.html', items=items, subject_areas=unique_subject_areas,
                               unique_years=unique_years, selected_sort=selected_sort,
                               selected_field=selected_field, selected_year=selected_year,
                               search_query=search_query)
    except Exception as e:
        print("Error browsing publications:", e)
        return render_template('404.html')


@app.route("/publish")
def publish():
    """Route for rendering the upload form"""
    return render_template("publish.html")

@app.route("/chatbot")
def chatbot():
    """Route for rendering the chatbot form"""
    return render_template("chatbot.html")


@app.route('/doc')
def doc():
    """Route for the documentation page"""
    return render_template('doc.html')

@app.route("/edit")
def edit():
    """Route for rendering the unapprove docx"""
    return render_template("edit.html")


@app.route("/api", methods=["POST"])
def api():
    """Route to handle POST requests for the chatbot API"""
    # Get the message from the POST request
    message = request.json.get("message")
    # Send the message to OpenAI's API and receive the response
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": message}
        ]
    )
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

    This route handles form submission from 'publish.html' or any other page.
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
    filename = save_file(uploaded_file)

    if filename:
        if db_connection.insert_upload(title, authors, publicationDate, thesisAdvisor, department, degree, subjectArea, abstract, filename):
            # Insertion successful
            print("Upload record inserted successfully!")
            return "Upload successful!"
        else:
            # Insertion failed
            print("Failed to insert upload record.")

    return "Data submitted successfully!"


@app.route('/publication_detail/<int:item_id>')
def publication_detail(item_id):
    """
    Route for displaying publication details.

    This route fetches the publication details from the database based on the item_id and displays them.
    """
    # Fetch the publication details from the database based on the item_id
    db_conn = DBConnection(DB)
    item = db_conn.get_publication_by_id(item_id)
    db_conn.close_connection()

    if item is not None:
        # Read the text content of the PDF file using PyPDF2
        pdf_path = os.path.join('uploads', item['file_path'])
        text_content = read_pdf_text(pdf_path)

        return render_template('publication_detail.html', item=item, text_content=text_content)
    else:
        # Handle the case where the publication with the specified ID is not found
        return render_template('404.html'), 404


def read_pdf_text(pdf_path):
    """
    Function to read the text content of a PDF file using PyPDF2.

    This function reads the text content of a PDF file and returns it as a string.
    """
    text_content = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
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
    pdf_path = os.path.join('uploads', filename)
    return send_file(pdf_path, mimetype='application/pdf')

def extract_sections_from_text(text):
    # Load the spaCy NLP model for English (or your desired language)
    nlp = spacy.load("en_core_web_sm")

    # Process the extracted text using spaCy to extract sections
    doc = nlp(text)

    # Create a dictionary to store the extracted sections
    sections = {}

    # Define the section names you want to extract (e.g., "Introduction", "Methods", etc.)
    section_names = ["Introduction", "Methods", "Results", "Discussion"]

    # Loop through each sentence in the document
    for sentence in doc.sents:
        # Check if the sentence starts with one of the section names
        for section_name in section_names:
            if sentence.text.startswith(section_name):
                # Store the section content in the dictionary
                sections[section_name] = sentence.text
                break

    return sections

def extract_text_from_pdf(file_path):
    # Load the spaCy NLP model for English (or your desired language)
    nlp = spacy.load("en_core_web_sm")

    with open(file_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfFileReader(pdf_file)
        text = ""
        for page_num in range(pdf_reader.getNumPages()):
            page = pdf_reader.getPage(page_num)
            text += page.extract_text()

        # Process the extracted text using spaCy to enhance extraction quality
        doc = nlp(text)
        return doc.text


def convert_to_imrad(file_path):
    # Load the spaCy NLP model for text extraction
    nlp = spacy.load("en_core_web_sm")

    with open(file_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfFileReader(pdf_file)
        text = ""
        for page_num in range(pdf_reader.getNumPages()):
            page = pdf_reader.getPage(page_num)
            text += page.extract_text()

        # Process the extracted text using spaCy to enhance extraction quality
        doc = nlp(text)

        # Define a mapping of section names to their corresponding regular expressions
        section_patterns = {
            "Introduction": r"Introduction[:\n\s]*(.+?)\n(.*?)(?=\b(?:Methods|Results|Discussion)\b|$)",
            "Methods": r"Methods[:\n\s]*(.+?)\n(.*?)(?=\b(?:Results|Discussion)\b|$)",
            "Results": r"Results[:\n\s]*(.+?)\n(.*?)(?=\b(?:Discussion)\b|$)",
            "Discussion": r"Discussion[:\n\s]*(.+?)\n(.+)",
        }

        # Read the PDF content
        text_content = extract_text_from_pdf(file_path)

        # Extract sections using spaCy
        sections = extract_sections_from_text(text_content)

        # Create a new PDF with the extracted content in IMRAD format
        output_file_path = file_path.replace(os.path.splitext(file_path)[1], '_imrad.pdf')

        doc = SimpleDocTemplate(output_file_path, pagesize=letter)
        styles = getSampleStyleSheet()

        story = []
        for section_name, section_content in sections.items():
            # Create a Paragraph for the section name (in bold)
            section_name_paragraph = Paragraph(f"<b>{section_name}</b>", styles['Heading1'])
            story.append(section_name_paragraph)

            # Create a Paragraph for the section content
            section_content_paragraph = Paragraph(section_content, styles['Normal'])
            story.append(section_content_paragraph)

            # Add a Spacer (optional) to separate sections
            story.append(Spacer(1, 20))

        # Build the PDF
        doc.build(story)

        return output_file_path



@app.route('/convert_to_imrad/<int:item_id>', methods=['GET'])
def convert_to_imrad_route(item_id):
    """
    Route to convert a publication to IMRAD format.

    This route takes the item_id of a publication, converts it to IMRAD format, and returns the converted PDF.
    """
    publication = db_connection.get_publication_by_id(item_id)
    if publication is None:
        return jsonify({"message": "Publication not found."}), 404

    file_name = os.path.basename(publication['file_path'])
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    try:
        # Perform the conversion and get the converted file path
        converted_file_path = convert_to_imrad(file_path)

        # Update the database with the converted file path
        db_connection.update_converted_file_path(item_id, converted_file_path)

        # Return the converted PDF for preview in the page
        return send_file(converted_file_path, as_attachment=False)
    except Exception as e:
        return jsonify({"message": "Error converting to IMRAD.", "error": str(e)}), 500


def generate_apa_citation_from_data(publication):
    """
    Function to generate APA citation from publication data.

    This function takes the publication data and returns the APA citation in the specified format.
    """
    # Format the authors' names
    authors = publication['authors'].split('; ')
    num_authors = len(authors)

    if num_authors == 1:
        formatted_authors = authors[0].split()[-1]
    elif num_authors == 2:
        formatted_authors = f"{authors[0].split()[-1]} & {authors[1].split()[-1]}"
    else:
        formatted_authors = ", ".join(author.split()[-1] for author in authors[:-1])
        formatted_authors += f", & {authors[-1].split()[-1]}"

    # Format the APA citation based on the publication data
    apa_citation = f"{formatted_authors}. ({publication['year']}). {publication['title']}. {publication['thesisAdvisor']}. {publication['department']}. {publication['degree']}."

    return apa_citation


@app.route('/generate_apa_citation/<int:item_id>')
def generate_apa_citation(item_id):
    """
    Route to generate APA citation for a publication.

    This route takes the item_id of a publication, generates the APA citation, and returns it as a JSON response.
    """
    db_conn = DBConnection(DB)
    # Fetch the publication by its ID from the database
    publication = db_conn.get_publication_by_id(item_id)

    if publication:
        # Generate the APA citation based on the publication data
        apa_citation = generate_apa_citation_from_data(publication)

        # Return the APA citation as a JSON response
        return jsonify({"apa_citation": apa_citation})

    return jsonify({"error": "Publication not found"})


if __name__ == '__main__':
    app.run()
