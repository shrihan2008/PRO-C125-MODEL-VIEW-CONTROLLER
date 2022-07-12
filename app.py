from flask import Flask,jsonify,request
from  classifier import get_prediction

app=Flask(__name__)

@app.route('/predict-letters',methods=["POST"])


def predict_data():
    image=request.files.get("letters")
    prediction=get_prediction(image)
    return jsonify({
        'prediction':prediction
    }),890

if __name__=='__main__':
    app.run(debug=True)
