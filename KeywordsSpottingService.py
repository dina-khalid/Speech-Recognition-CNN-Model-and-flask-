import tensorflow.keras as keras
import numpy as np
import librosa


modelPath = "model.h5"
numSampleToConsider=22050 #1 sec
class _KeywordsSpottingService:

    model=None
    _mappings=[
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
    ]
    _instance= None
    
    def predict(self, filePath):
        #extract MFCCS
        MFCCs=self.preprocess(filePath) #(#segments,#coef)

        #convert 2D array to 4d array  #(#samples,#segments, #coef, #channels=1 )
        MFCCs=MFCCs[np.newaxis, ...,np.newaxis]
        #make prediction
        predictions=self.model.predict(MFCCs)
        predicted_index=np.argmax(predictions)
        predicted_keyword=self._mappings[predicted_index]
        return  predicted_keyword




    def preprocess(self, filePath, nMFCCs=13, nFTT=2048, hopLength=512):
        signal, sample_rate = librosa.load(filePath)

        if len(signal) >= numSampleToConsider:
            # ensure consistency of the length of the signal
            signal = signal[:numSampleToConsider]

            # extract MFCCs
            MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=nMFCCs, n_fft=nFTT,
                                         hop_length=hopLength)
        return MFCCs.T




def KeywordsSpottingService():
    #ensure we have 1 instance of kss only
    if _KeywordsSpottingService._instance is None:
        _KeywordsSpottingService._instance=_KeywordsSpottingService()
        _KeywordsSpottingService.model=keras.models.load_model(modelPath)
    return _KeywordsSpottingService._instance


if __name__=="__main__":
    kss= KeywordsSpottingService()

    word1 = kss.predict("Test/Right.wav")
    word2= kss.predict("Test/No.wav")

    print(f"Predicted words {word1}, {word2}")
