import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
from algorithms import *

def read_data(trainfile='MNIST_train.csv', validationfile='MNIST_validation.csv'):

    dftrain = pd.read_csv(trainfile)
    dfval = pd.read_csv(validationfile)

    dftrain.drop('even', axis = 1, inplace = True)
    dfval.drop('even', axis = 1, inplace = True)

    featurecols = list(dftrain.columns)
    #print(featurecols)
    featurecols.remove('label')
    #featurecols.remove('even')
    targetcol = 'label'

    Xtrain = dftrain[featurecols]
    ytrain = dftrain[targetcol]

    Xval = dfval[featurecols]
    yval = dfval[targetcol]

    return (Xtrain, ytrain, Xval, yval)

if __name__ == "__main__":
    Xtrain, ytrain, Xval, yval = read_data('MNIST_train.csv', 'MNIST_validation.csv')
    
    Xtrain = np.array(Xtrain)
    ytrain = np.array(ytrain)
    Xval = np.array(Xval)
    yval = np.array(yval)

    # Define base models
    pipeline_lr = MyPipeline([
        ('scaler', MyStandardScaler()),
        ('lr', MyLogisticRegression(learning_rate=0.01, n_iterations=10, batch_size=32))])
    
    pipeline_svm = MyPipeline([
        ('scaler', MyStandardScaler()),
        ('svm', MultiClassSVM(n_iters = 20, learning_rate=0.001))])
   
    pipeline_xgb = MyPipeline([
        ('scaler', MyStandardScaler()),
        ('xgb', XGBoostClassifier(n_estimators= 12, max_depth = 6, learning_rate=0.3745, colsample_bytree=0.05, reg_lambda= 1))
    ])
    
    base_models = [pipeline_lr,pipeline_svm, pipeline_xgb]
    
    # Define meta-model
    meta_model = XGBoostClassifier(n_estimators= 5, max_depth = 4, learning_rate=0.3745, colsample_bytree= 1, reg_lambda= 1)
    
    # Create and train stacking classifier
    stacker = MyStackingClassifier(
        base_models=base_models,
        meta_model=meta_model,
        k=2
    )
    stacker.fit(Xtrain, ytrain)
    
    # Create inference model
    inference_model = StackingInferenceModel(stacker)
    
    y_pred = inference_model.predict(Xval)
    
    score = f1_score(yval, y_pred, average='weighted')
    print(f"\nStacking Model Accuracy: {score:.4f}")
