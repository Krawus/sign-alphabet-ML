import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC



if __name__ == "__main__":
    #collect arguments from user and convert it into paths to the files
    parser = argparse.ArgumentParser()
    parser.add_argument('inputData', type=str)
    parser.add_argument('predictionsFile', type=str)
    args = parser.parse_args()

    inputData = Path(args.inputData)
    predictionsFile = Path(args.predictionsFile)

    #read file passed as argument into pandas dataframe
    df = pd.read_csv(inputData)
    #get only landmarks data to predict letter
    X = df.iloc[:, 1:64]
    #the model will also use handedness label to make letter prediction
    X['hand'] = df['handedness.label']
    # encode hand string -> "Left" is 0, "Right" is 1
    handLabelEncoder = joblib.load("models/hand_label_encoder.pkl")
    X['hand'] = handLabelEncoder.transform(X['hand'])

    #load the trained model - SVM with poly degree=2 kernel and other parameteres found using GridSearch
    SVCmodel = joblib.load("models/SVCmodel.pkl")
    #predict letter based on landmarks and hand
    predictions = SVCmodel.predict(X)

    #save predictions into the file, where every predicted letter is printed in new line
    with open(predictionsFile, mode='w', encoding='utf-8') as file:
        for prediction in predictions:
            file.write(f'{prediction}\n')
