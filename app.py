"""
This module contains the Flask application for your project.

It handles routes, templates, and database connections.
"""
import os
import secrets
import openai
import spacy
import re
import io
import torch
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from PyPDF2 import PdfFileReader, PdfFileWriter
from flask import redirect, url_for
from PyPDF2 import PageObject
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from docx import Document
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from transformers import BertTokenizer, BertForSequenceClassification
import PyPDF2
from flask import Flask, render_template, request, jsonify, session, send_file
from db_connection import DBConnection

# Mapping of section indices to section names
section_names = {0: "Introduction", 1: "Method", 2: "Result", 3: "Discussion"}

app = Flask(__name__)

# Generate a secure secret key for the session
app.secret_key = secrets.token_hex(16)
# Load the spaCy NLP model for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")
DB = 'database.db'

# Load the BERT model and tokenizer
model_path = "Models\BERT Model" 
tokenizer = BertTokenizer.from_pretrained(model_path)
bert_model = BertForSequenceClassification.from_pretrained(model_path)

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


@app.route('/DV')
def DV():
    """Route for the documentation page"""
    return render_template('DV.html')

@app.route("/research")
def research():
    """Route for rendering the unapprove docx"""
    try:
        db_conn = DBConnection('database.db')

        selected_sort = request.args.get('sort', 'latest')
        selected_field = request.args.get('field', '')
        selected_year = request.args.get('year', '')
        search_query = request.args.get('search', '')  # Get the search query from the query parameters

        items, unique_subject_areas, unique_years = db_conn.fetch_research_publications(
            selected_sort, selected_field, selected_year, search_query)

        db_conn.close_connection()

        return render_template('research.html', items=items, subject_areas=unique_subject_areas,
                               unique_years=unique_years, selected_sort=selected_sort,
                               selected_field=selected_field, selected_year=selected_year,
                               search_query=search_query)
    except Exception as e:
        print("Error browsing publications:", e)
        return render_template('404.html')



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
    status = request.form['status']  # Extract status from form data
    uploaded_file = request.files['file']
    filename = save_file(uploaded_file)

    if filename:
        if db_connection.insert_upload(title, authors, publicationDate, thesisAdvisor, department, degree, subjectArea, abstract, filename, status):
            print("Upload record inserted successfully!")
            if status == "Working":
                return redirect(url_for('abstract', title=title))  # Redirect to 'abstract' route with title
            else:
                return "Upload successful!"
        else:
            print("Failed to insert upload record.")

    return "Data submitted successfully!"


@app.route('/abstract')
def abstract():
    record = db_connection.get_last_unapproved()
    if record is not None:
        title = record['title']
        # ... extract other data from record ...
        return render_template('abstract.html', title=title)  # Pass data to template
    else:
        return "No unapproved records found."

@app.route('/generate_abstract')
def generate_abstract():
    print("Starting to generate abstract...")
    # Get the last unapproved record
    record = db_connection.get_last_unapproved()
    if record is not None:
        print("Record found, processing...")
        title = record['title']
        file_path = 'uploads/' + record['file_path']    # Get the file path from the record

        # Open the .docx file and extract the text
        doc = Document(file_path)  # Use the file path to open the .docx file
        doc_text = " ".join([p.text for p in doc.paragraphs])

        chunk_size = 512
        text_chunks = [doc_text[i:i+chunk_size] for i in range(0, len(doc_text), chunk_size)]
        section_texts = {section: "" for section in section_names.values()}

        # Define batch size
        batch_size = 10

        # Process text in batches
        for i in range(0, len(text_chunks), batch_size):
            print(f"Processing batch {i//batch_size + 1}...")
            batch = text_chunks[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            
            with torch.no_grad():
                outputs = bert_model(**inputs)
                predicted_sections = torch.argmax(outputs.logits, dim=1)
                
                
            for j, input_ids in enumerate(inputs["input_ids"]):
                current_section = section_names[predicted_sections[j].item()]
                for token in input_ids:
                    token_text = tokenizer.decode([token.item()], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    
                    if current_section is not None:
                        if section_texts[current_section] and not section_texts[current_section].endswith(' '):
                            section_texts[current_section] += ' '
                        
                        section_texts[current_section] += token_text
                    print(f"output: {section_texts}")

            print(f"Finished processing batch {i//batch_size + 1}.")

        print("Finished processing. Sending response...")
        return jsonify(section_texts)
    else:
        print("No unapproved records found.")
        return jsonify("No unapproved records found.")




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

def simpleSplit(text, fontName, fontSize, maxWidth):
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
    # Remove non-alphanumeric characters
    text = re.sub(r'\W+', ' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    text = [word for word in word_tokens if not word in stop_words]
    
    return ' '.join(text)

def convert_to_imrad(file_path):
    try:
        with open(file_path, "rb") as f:
            pdf_reader = PdfFileReader(f)
            pdf_text = ""
            
            for page in pdf_reader.pages:
                pdf_text += page.extract_text()

        # Clean the extracted text
        pdf_text = clean_text(pdf_text)

        chunk_size = 512
        text_chunks = [pdf_text[i:i+chunk_size] for i in range(0, len(pdf_text), chunk_size)]
        section_texts = {section: "" for section in section_names.values()}

        # Define batch size
        batch_size = 10

        # Process text in batches
        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            
            with torch.no_grad():
                outputs = bert_model(**inputs)
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



# Route to convert a publication to IMRAD format
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
    


if __name__ == '__main__':
    app.run()