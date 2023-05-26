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
    # encode hand string -> "Left" is 0, "Right" is 1
    X['hand'] = df['handedness.label']
    le = LabelEncoder()
    X['hand'] = le.fit_transform(X['hand'])


    ## TODO @@@@@@@@ TO DELETE WHEN DEPLOY @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # y = df.letter
    
    # X, _, y, _ = train_test_split(X, y, stratify=y, random_state=12, test_size=0.8)

    ## TODO @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    #load the trained model
    SVCmodel = joblib.load("SVCmodel.pkl")
    #predict hand based on landmarks
    predictions = SVCmodel.predict(X)

    #save predictions into the file
    with open(predictionsFile, mode='w', encoding='utf-8') as file:
        for prediction in predictions:
            file.write(f'{prediction}\n')

