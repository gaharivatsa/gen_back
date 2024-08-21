from dotenv import load_dotenv
import os
from flask import Flask, request, jsonify
from tika import parser
from flask_cors import CORS
import google.generativeai as genai
import json

load_dotenv()

api_key = os.getenv("GENAI_API_KEY")
genai.configure(api_key=api_key)

model = genai.GenerativeModel(model_name="gemini-1.5-flash")

app = Flask(__name__)

CORS(app, origins=["https://ampli5.vercel.app/"])

@app.route('/resume_enhance', methods=['POST'])
def process_pdf():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        
        file = request.files['file']
        
        resume_raw = parser.from_buffer(file.read())
        resume_text = resume_raw.get('content', '')

        if not resume_text:
            return jsonify({"error": "Failed to extract text from PDF"}), 400

        # Generate content using the generative AI model
        response = model.generate_content([
            """Analyze the resume text provided below, focusing on three key areas: 
            1) Highlighting the candidate's strengths relevant to a Data Analyst role, 
            2) Identifying areas for improvement, particularly in project descriptions and skills alignment, 
            3) Providing specific, actionable suggestions for improving the resume. 
            Use this schema: {
              "type": "object",
              "properties": {
                "strengths": {
                  "type": "array",
                  "items": {
                    "type": "object",
                    "properties": {
                      "description": { "type": "string" },
                      "evidence": {
                        "type": "array",
                        "items": { "type": "string" }
                      }
                    },
                    "required": ["description", "evidence"]
                  }
                },
                "areas_for_improvement": {
                  "type": "array",
                  "items": {
                    "type": "object",
                    "properties": {
                      "description": { "type": "string" },
                      "suggestions": {
                        "type": "array",
                        "items": { "type": "string" }
                      }
                    },
                    "required": ["description", "suggestions"]
                  }
                },
                "actionable_suggestions": {
                  "type": "array",
                  "items": { "type": "string" }
                }
              },
              "required": ["strengths", "areas_for_improvement", "actionable_suggestions"]
            }
            Resume text caution: don't use ** for indexing give the information in json format: """ + resume_text
        ])

        # Extract and clean up the response
        generated_text = response.text

        # Remove the code block formatting
        if generated_text.startswith("```json"):
            generated_text = generated_text.strip("```json\n").strip("```")

        # Convert the cleaned string back to a JSON object
        json_response = json.loads(generated_text)

        # Return the AI-generated analysis as a JSON object
        return jsonify(json_response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/compare', methods=['POST'])
def compare():
    try:
        # Ensure both 'resume' and 'jd' files are provided
        if 'resume' not in request.files or 'jd' not in request.files:
            return jsonify({"error": "Both resume and job description files must be provided"}), 400
        
        # Extract text from the resume
        resume_file = request.files['resume']
        resume_raw = parser.from_buffer(resume_file.read())
        resume_text = resume_raw.get('content', '')

        if not resume_text:
            return jsonify({"error": "Failed to extract text from resume PDF"}), 400

        # Extract text from the job description
        jd_file = request.files['jd']
        jd_raw = parser.from_buffer(jd_file.read())
        jd_text = jd_raw.get('content', '')

        if not jd_text:
            return jsonify({"error": "Failed to extract text from job description PDF"}), 400

        # Preprocess the texts (you can add more preprocessing steps as needed)
        response = model.generate_content([f""" 
            Analyze the following Job Description (JD) and Resume, then provide a JSON-formatted response with the following structure:
            {{
                "similarity_score": <calculated similarity score as a decimal between 0 and 1>,
                "content": {{
                    "improvement_suggestions": ["list of specific suggestions to improve the resume"],
                    "strengths": ["list of strengths identified in the resume"],
                    "areas_for_improvement": ["list of areas for improvement in the resume"]
                }}
            }}
            Job Description (JD): "{jd_text}"
            Resume: "{resume_text}"
        """])
        generated_text = response.text

        # Remove the code block formatting
        if generated_text.startswith("```json"):
            generated_text = generated_text.strip("```json\n").strip("```")

        # Convert the cleaned string back to a JSON object
        json_response = json.loads(generated_text)

        # Return the AI-generated analysis as a JSON object
        return jsonify(json_response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)
