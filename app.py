from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# load trained model
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["text"]

    # convert text → numbers
    data = vectorizer.transform([text])

    # predict
    prediction = model.predict(data)[0]

    # mapping (CORRECT)
    if prediction == 2:
        result = "Positive 😊"
    elif prediction == 1:
        result = "Neutral 😐"
    else:
        result = "Negative 😡"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
