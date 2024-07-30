from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from sentence_transformers import SentenceTransformer




app = Flask(__name__)

@app.route('/model_endpoint', methods=['POST'])
def handle_post():
    # Get the data sent in the POST request
    data = request.get_json()

    
    sentence_transformers = SentenceTransformer('all-MiniLM-L6-v2')


    ideal_answer = data.ideal_answer
    student_answer = data.student_answer

    e1 = sentence_transformers.encode([ideal_answer])
    e2 = sentence_transformers.encode([student_answer])

    siamese_model = load_model('model/siamese_model.h5')




    prediction = siamese_model.predict([e1, e2])

    

    # Process the data (this is where your logic would go)
    response = {
        'prediction': prediction
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
