# app.py

from flask import Flask, render_template, request
from translator import translate

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    translated = ""
    if request.method == "POST":
        text = request.form["text"]
        translated = translate(text)
    return render_template("index.html", translation=translated)

if __name__ == "__main__":
    app.run(debug=True)
