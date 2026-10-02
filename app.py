from flask import Flask, request, jsonify
from flask_cors import CORS

import os
import traceback

from predict import predict_emotion


app = Flask(__name__)

CORS(app)


UPLOAD_FOLDER = "uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


@app.route("/")
def home():

    return jsonify({
        "message": "Speech Emotion Recognition API",
        "status": "running"
    })


@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        if "audio" not in request.files:

            return jsonify({
                "error": "No audio file provided"
            }), 400


        audio_file = request.files[
            "audio"
        ]


        if audio_file.filename == "":

            return jsonify({
                "error": "No file selected"
            }), 400


        filename = (
            "prediction_audio.wav"
        )


        file_path = os.path.join(
            UPLOAD_FOLDER,
            filename
        )


        audio_file.save(
            file_path
        )


        result = predict_emotion(
            file_path
        )


        return jsonify(
            result
        )


    except Exception as e:

        traceback.print_exc()

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )