import random,os
from flask import Flask, request, jsonify
from KeywordsSpottingService import KeywordsSpottingService


app = Flask(__name__)


@app.route("/predict",methods=["POST"])

def predict():

    #get and save audio file
    audioFile=request.files["file"]
    fileName=str(random.randint(0,100000))
    audioFile.save(fileName)

    #invoke ketwordspottingservice
    kss=KeywordsSpottingService()

    #make a prediction
    predictedWord=kss.predict(fileName)

    #remove the audio file
    os.remove(fileName)

    #sendback the predicted file in json format
    data={"keyWord":predictedWord}

    return jsonify(data)

if __name__=="__main__":
    app.run(debug=False)


