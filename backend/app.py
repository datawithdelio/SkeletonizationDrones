from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tempfile
from skeleton_generation.skel import skeletonize_video, skeletonize_img
import os
import uuid
import json
from skeleton_generation.openai_creation import save_generation, describe_image

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:5173"}}, supports_credentials=True)

frontend_folder = os.path.join(os.getcwd(), "..", "frontend")
dist_folder = os.path.join(frontend_folder, "dist")

@app.route('/', defaults={"filename": ""})
@app.route('/<path:filename>')
def index(filename):
    return send_from_directory(dist_folder, filename or "index.html")

file_dict = {}

base_dir = os.path.dirname(os.path.abspath(__file__))
app.config['RESULT_FOLDER'] = os.path.join(base_dir, 'output_path')
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

@app.route('/api/upload', methods=["POST"])
def upload():
    if 'file' not in request.files:
        return jsonify({"msg": "File Not Found!"}), 400

    file = request.files['file']
    generation_settings = json.loads(request.form.get('generationSettings', '{}'))
    file_unique_id = str(uuid.uuid4())

    with tempfile.NamedTemporaryFile(delete=False) as input_file:
        input_path = input_file.name
        file.save(input_path)

    file_name, ext = os.path.splitext(file.filename)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']

    if ext.lower() in image_extensions:
        skeleton_fil_name = f"{file_name}-skeleton.png"
        skeleton_data_name = f"{file_name}-data.pt"
        skeletonize_img(input_path, app.config['RESULT_FOLDER'], skeleton_fil_name, skeleton_data_name, generation_settings)

        # ✅ Check if skeleton image was created before describing
        skeleton_path = os.path.join(app.config['RESULT_FOLDER'], skeleton_fil_name)
        if os.path.exists(skeleton_path):
            caption = describe_image(skeleton_path)
        else:
            caption = "Unable to generate description — skeleton output not found."

    elif ext.lower() in video_extensions:
        skeleton_fil_name = f"{file_name}-skeleton.mp4"
        skeleton_data_name = f"{file_name}-data.pt"
        skeletonize_video(input_path, app.config['RESULT_FOLDER'], skeleton_fil_name, skeleton_data_name, generation_settings)
        caption = "Video processed — no caption generated."

    else:
        return jsonify({"msg": "Invalid file type!"}), 400

    file_dict[file_unique_id] = {
        "skeleton": skeleton_fil_name,
        "point_data": skeleton_data_name
    }

    os.remove(input_path)

    return jsonify({
        "msg": "Upload Successful!",
        "id": file_unique_id,
        "caption": caption
    }), 200

@app.route('/api/openai/upload', methods=['POST'])
def openai_upload():
    prompt = request.json.get('prompt')
    generation_settings = request.json.get('generationSettings')
    file_unique_id = str(uuid.uuid4())

    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as input_file:
        input_path = input_file.name
        save_generation(prompt, input_path)

    skeleton_fil_name = f"{file_unique_id}-skeleton.png"
    skeleton_data_name = f"{file_unique_id}-data.pt"
    skeletonize_img(input_path, app.config['RESULT_FOLDER'], skeleton_fil_name, skeleton_data_name, generation_settings)

    # ✅ Confirm the file exists before generating a caption
    skeleton_path = os.path.join(app.config['RESULT_FOLDER'], skeleton_fil_name)
    if os.path.exists(skeleton_path):
        caption = describe_image(skeleton_path)
    else:
        caption = "Skeleton image not found."

    file_dict[file_unique_id] = {"skeleton": skeleton_fil_name}
    os.remove(input_path)

    return jsonify({
        "msg": "AI Generation Successful!",
        "id": file_unique_id,
        "caption": caption
    }), 200

@app.route('/api/view/<unique_id>', methods=['GET'])
def view(unique_id):
    if unique_id in file_dict:
        file_name = file_dict[unique_id]["skeleton"]
        if os.path.exists(os.path.join(app.config['RESULT_FOLDER'], file_name)):
            return send_from_directory(app.config['RESULT_FOLDER'], file_name, mimetype='video/mp4')
    return jsonify({"msg": "File or ID Not Found!"}), 404

@app.route('/api/download/<unique_id>', methods=["GET"])
def download(unique_id):
    if unique_id in file_dict:
        file_name = file_dict[unique_id]["skeleton"]
        if os.path.exists(os.path.join(app.config['RESULT_FOLDER'], file_name)):
            return send_from_directory(app.config['RESULT_FOLDER'], file_name, as_attachment=True, mimetype='video/mp4')
    return jsonify({"msg": "File Not Found!"}), 404

@app.route('/api/download_data/<unique_id>', methods=["GET"])
def download_data(unique_id):
    if unique_id in file_dict:
        file_name = file_dict[unique_id].get("point_data")
        if file_name and os.path.exists(os.path.join(app.config['RESULT_FOLDER'], file_name)):
            return send_from_directory(app.config['RESULT_FOLDER'], file_name, as_attachment=True)
    return jsonify({"msg": "Data File Not Found!"}), 404

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
