from flask import Flask, render_template, request
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Create Flask app
app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("sentiment_model.keras")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 200

@app.route("/", methods=["GET", "POST"])
def home():

    sentiment = ""
    confidence = ""

    if request.method == "POST":

        review = request.form["review"]

        # Convert text to sequence
        seq = tokenizer.texts_to_sequences([review])

        # Pad sequence
        padded = pad_sequences(
            seq,
            maxlen=MAX_LEN,
            padding="post",
            truncating="post"
        )

        # Prediction
        prediction = model.predict(padded, verbose=0)[0][0]

        if prediction >= 0.5:
            sentiment = "Positive 😊"
            confidence = round(prediction * 100, 2)
        else:
            sentiment = "Negative 😞"
            confidence = round((1 - prediction) * 100, 2)

    return render_template(
        "index.html",
        sentiment=sentiment,
        confidence=confidence
    )

if __name__ == "__main__":
    app.run(debug=True)
