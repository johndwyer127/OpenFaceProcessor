from sklearn.externals import joblib
import sys
import numpy as np

def main():

    model = joblib.load('test.joblib') #load sci kit model

    args = sys.argv #reads command line arguments
    args.pop(0) #removes name of script
    argsArray = np.array(args) #convert to array for passing as features
    argsArray.reshape(1,-1) #convert to correct shape
    argsArray = argsArray.astype(np.float64) #convert to float64

    pred = model.predict([argsArray])  #make prediction from arguments
    return pred[0] #return prediction

main()
