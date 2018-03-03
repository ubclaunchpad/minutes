from flask import Flask, request, jsonify


app = Flask(__name__)


@app.route("/")
def hello():
    return "Hello, world!"


@app.route("/upload", methods=['POST'])
def upload_audio():
    """
    Accepts a file as part of a POST request.
    """

    if 'file' not in request.files:
        return "No file given in request.", 400

    # TODO: Process the received audio data!
    f = request.files['file']
    data = f.read()

    return jsonify({
        'mimetype': f.mimetype,
        'filename': f.filename,
        'leading_bytes': str(data)[:64],
        'size': len(data),
    })


if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=80)
