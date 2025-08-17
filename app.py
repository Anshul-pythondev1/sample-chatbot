import os
from flask import Flask , render_template, request, jsonify
from werkzeug.utils import secure_filename
from ragpipeline import process_pdf_and_store , query_rag
from dotenv import load_dotenv
import google.generativeai as genai
import pinecone
load_dotenv()


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not PINECONE_API_KEY or not GEMINI_API_KEY:
    raise ValueError("Missing API keys! Please check your .env file.")

app = Flask(__name__)
app.config['UPLOAD FOLDER'] = "uploads"
os.makedirs(app.config['UPLOAD FOLDER'],exist_ok=True)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/upload",methods = ["POST"])
def upload_files():
    if "file" not in request.files:
        return jsonify({"Error" : "The file is not given"}),400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"Error" : "File not selected"}),400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD FOLDER"],filename)
    file.save(filepath)

    process_pdf_and_store(filepath)
    return jsonify({"Message" : "File processed and stored successfully"})

@app.route("/chat" , methods = ["POST"])
def chat():
    user_question = request.json.get("question")
    if not user_question:
        return jsonify({"Error" : "Question not found."}), 400
    
    answer = query_rag(user_question)
    return jsonify({"answer" : answer})

if __name__ == "__main__":
    app.run(debug=True)
    
    
