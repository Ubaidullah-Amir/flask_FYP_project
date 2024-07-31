from flask import Flask, request, jsonify
from keras.models import load_model
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

from keras import backend as K

# Define the custom lambda function
def custom_lambda(tensors):
    return K.abs(tensors[0] - tensors[1])

# Load models and encoders once at startup
siamese_model = load_model('model/siamese_model.h5', custom_objects={'<lambda>': custom_lambda})

sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
@app.route('/', methods=['GET'])
def handle_get():
    
    response = {
        'message': "hello"
    }

    return jsonify(response)


@app.route('/model_endpoint', methods=['POST'])
def handle_post():
    # Get the data sent in the POST request
    data = request.get_json()

    ideal_answer = data.get('ideal_answer')
    student_answer = data.get('student_answer')

    # Encode the sentences
    e1 = sentence_transformer.encode([ideal_answer])
    e2 = sentence_transformer.encode([student_answer])

    # Make a prediction using the loaded model
    prediction = siamese_model.predict([e1, e2]).tolist()
 
    # prediction = 3 
    # Prepare the response
    response = {
        'prediction': prediction
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
