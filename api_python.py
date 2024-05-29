import re
import pickle
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load the tokenizer and model
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

model = load_model('cnn_lstm_model.h5')

# Define the ArabicCleaning class for text preprocessing
class ArabicCleaning():
    def __init__(self):
        pass

    def clean(self, text):
        arabic_pattern = r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]'
        arabic_pattern_others = r'[^\w\s\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]'

        def remove_special_words(text):
            words = text.split()
            text = [word for word in words if '#' not in word and '_' not in word]
            text = ' '.join(text)
            return text

        def keep_only_arabic_letters(text):
            words = text.split()
            processed_words = []
            for word in words:
                arabic_letters_only = ''.join([char for char in word if re.match(arabic_pattern, char) and char not in ["؟", "؛", "،"]])
                processed_words.append(arabic_letters_only)
            return ' '.join(processed_words)

        def check_empty(text):
            if len(text.split()) == 0:
                return ''
            else:
                return text

        text = remove_special_words(text)
        text = re.sub(arabic_pattern_others, '', text)
        text = re.sub(r'[0-9]', '', text)
        text = re.sub(r'[a-zA-Z]', '', text)
        text = keep_only_arabic_letters(text)
        text = check_empty(text)

        return text

# Define a route for the Flask app with a URL prefix
@app.route('/api/predict', methods=['POST'])
def predict():
    # Get the text from the request
    text = request.json['text']

    # Clean and preprocess the text
    clean_text = ArabicCleaning().clean(text)
    sequences = tokenizer.texts_to_sequences([clean_text])
    padded_sequences = pad_sequences(sequences, maxlen=65)

    # Predict the probabilities
    prediction = model.predict(padded_sequences)

    # Get the class names and their corresponding probabilities
    class_names = encoder.classes_
    probabilities = prediction[0]*100
    dialect = encoder.inverse_transform([np.argmax(prediction)])
    
    # create a dictionary with the class names and their corresponding probabilities
    countries = {"EG": "Egypt", "LB": "Lebanon", "LY": "Libya", "MA": "Morocco", "SD": "Sudan"}
    probabilities_dict = {countries.get(key, key): float(value) for key, value in zip(class_names, probabilities)}
    
    # Get the predicted country
    predicted_country = countries[dialect[0]]
    
    # Add the predicted country to the probabilities dictionary
    probabilities_dict['predicted_country'] = predicted_country
    
    return jsonify(probabilities_dict)



if __name__ == '__main__':
    app.run(debug=True)


# run the following command in the terminal
# py -3.8 api_python.py