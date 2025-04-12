# app.py
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import PyPDF2
import os
import tempfile

app = Flask(__name__)

# Configure API key
API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyBnbRuvRY_4LxRSJt4wePIjcnz9AK7Eydo")  # Get from environment or use default
genai.configure(api_key=API_KEY)



# Sample patient data (can be replaced with actual database)
patient_profile = {
    "age": 45,
    "gender": "Male",
    "weight": "82kg",
    "known_conditions": ["Type 2 Diabetes", "Hypertension"],
    "recent_vitals": {
        "blood_pressure": "145/90",
        "glucose_level": "160 mg/dL",
        "heart_rate": "88 bpm"
    }
}

def format_patient_data(profile):
    vitals = profile["recent_vitals"]
    return f"""
Age: {profile['age']}
Gender: {profile['gender']}
Weight: {profile['weight']}
Known Conditions: {", ".join(profile['known_conditions'])}
Recent Vitals:
  - Blood Pressure: {vitals['blood_pressure']}
  - Glucose Level: {vitals['glucose_level']}
  - Heart Rate: {vitals['heart_rate']}
"""

def generate_health_response(query):
    prompt = f"""
    Interact with the patient as a human for non-medical questions.
    Answer the following health-related question based on the patient's profile and health data, if asked:
    Question: {query}
    Patient Profile:
    {format_patient_data(patient_profile)}
    Provide a concise answer focusing on:
    1. Direct response to the question
    2. Relevant health insights from the data
    3. Personalized recommendations if applicable
    4. Any necessary precautions or warnings
    """
    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-pro")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error occurred: {str(e)}"

def extract_text_from_pdfs(pdf_files):
    all_text = ""
    temp_dir = tempfile.mkdtemp()
    
    try:
        for pdf_file in pdf_files:
            temp_path = os.path.join(temp_dir, pdf_file.filename)
            pdf_file.save(temp_path)
            
            try:
                with open(temp_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        extracted_text = page.extract_text()
                        if extracted_text:
                            all_text += extracted_text + "\n"
            except Exception as e:
                print(f"Error reading {pdf_file.filename}: {e}")
                
    except Exception as e:
        print(f"Error processing files: {e}")
        
    finally:
        # Clean up temp files
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)
        
    return all_text

def analyze_medical_reports(extracted_text):
    if not extracted_text.strip():
        return "No text could be extracted from the uploaded PDF(s). Please ensure the files contain readable text."
    
    prompt = f"""
    You are a medical assistant AI. A patient has uploaded their medical report(s).
    Your job is to help them understand everything clearly.
    Here is the full content of their report(s):
    --- START REPORT TEXT ---
    {extracted_text}
    --- END REPORT TEXT ---
    Based on the report(s), provide a detailed but understandable summary for the patient including:
    1. What condition(s) they may have or are being tested for
    2. Likely causes or contributing factors
    3. Common medications or treatments and what they do
    4. Side effects the patient should watch out for
    5. Important precautions or advice they should follow
    Avoid technical jargon where possible. Be empathetic and clear.
    """
    
    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-pro")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error during analysis: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('message', '')
    response = generate_health_response(query)
    return jsonify({'response': response})

@app.route('/api/upload', methods=['POST'])
def upload_files():
    if 'files[]' not in request.files:
        return jsonify({'response': 'No files uploaded'})
    
    files = request.files.getlist('files[]')
    
    if not files or files[0].filename == '':
        return jsonify({'response': 'No files selected'})
    
    # Process only PDF files
    pdf_files = [f for f in files if f.filename.lower().endswith('.pdf')]
    
    if not pdf_files:
        return jsonify({'response': 'No PDF files were uploaded. Please upload PDF files only.'})
    
    extracted_text = extract_text_from_pdfs(pdf_files)
    analysis = analyze_medical_reports(extracted_text)
    
    return jsonify({'response': analysis})

@app.route('/api/reset', methods=['POST'])
def reset_chat():
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True)