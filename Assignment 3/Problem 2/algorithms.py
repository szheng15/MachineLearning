import numpy as np
import utilities as utils

class Regressor:
    """
    Generic regression interface; returns random regressor
    Random regressor randomly selects value between max and min in training set.
    """
    
    def __init__( self, params=None ):
        """ Params can contain any useful parameters for the algorithm """
        self.min = 0
        self.max = 1
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.min = np.amin(ytrain)
        self.max = np.amax(ytrain)

    def predict(self, Xtest):
        ytest = np.random.rand(Xtest.shape[0])*(self.max-self.min) + self.min
        return ytest

class MeanPredictor(Regressor):
    """
    Returns the average target value observed; a reasonable baseline
    """
    def __init__( self, params=None ):
        self.mean = None
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.mean = np.mean(ytrain)
        
    def predict(self, Xtest):
        return np.ones((Xtest.shape[0],))*self.mean

class FSLinearRegression(Regressor):
    """
    Linear Regression with feature selection
    """
    def __init__( self, params=None ):
        self.weights = None
        self.features = range(10)  
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        Xless = Xtrain[:,self.features]
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xless.T,Xless)), Xless.T),ytrain)
        
    def predict(self, Xtest):
        Xless = Xtest[:,self.features]        
        ytest = np.dot(Xless, self.weights)       
        return ytest

class RidgeRegression(Regressor):
    """
    Ridge Regression with feature selection
    """
    def __init__( self, params=None ):
        self.weights = None
        self.features = range(10)  # max 230 features
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        Xless = Xtrain[:,self.features]
        lam = 1000
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xless.T,Xless) + lam*np.identity(Xless.shape[1])), Xless.T),ytrain)
        
    def predict(self, Xtest):
        Xless = Xtest[:,self.features]        
        ytest = np.dot(Xless, self.weights)       
        return ytest
    
