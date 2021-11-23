# Speech Recognition CNN Model and flask frame

This project is about predicting 10 words <p> [
        "down",
        "go",
        "left",
        "no",
        "off",
        "on",
        "right",
        "stop",
        "up",
        "yes"
    ] <p> from an audio file using CNN and flask
### The project consists of 5 classes:
- ### PrepareDataset 
    Here I extracted MFCCs from the dataset and saves them into a json file
- ### Train
    Load the dataset from json file split it then build the CNN <p>
    the model consist of 3 Conv layers each of them has Conv2D, BatchNormalization and MaxPooling2D 
    Then flatten the output and activate with softmax. Train the model and save it
- ### KeywordSpottingService
  - A single term class to load the model only once
  
- ### Server
  - Get and save audio file
  - Invoke ketword spotting service
  - Make a prediction and send it back in json format
  
- ### Client
   - Open the files
   - Package stuff to send and perform POST request
  
