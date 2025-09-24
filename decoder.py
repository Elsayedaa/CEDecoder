## Version 8
## Changes:

## Reworked timepoint embedding: 
## - Hybrid embedding removed
## - Feature embedding can now be done for a particular lag as well as the full time span
## - Sample embedding is not changed.

## Using the embedding strategies:
## - Instructions for using the embedding strategies are now in the documentation and docstrings.

import copy
import numpy as np
import sparse
from multiprocessing import Process, Manager
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from scipy.linalg import toeplitz

class decoder:
    def __init__(self, X, Y, model: str, params = None, boost = False, labels = None):
        """
        X: Must be a 3D matrix of shape (N(Neurons), N(Trials), N(Timepoints))
        Y: Must be a 1D array of categorical labels corresponding to the stimulus shown at each trial. 
        model: Must be an instance of a sklearn classification model
        params: Must be a dictionary of hyperparameters to be optimized
        boost: Boolean, default = False
        labels: If dependent variables are one-hot encoded, pass an array of the corresponding labels.
        """
        self.X = X
        self.Y = Y
        self.n_neurons = X.shape[0]
        self.Ydim = len(Y.shape)
        self.model = model
        self.params = params
        self.boost = boost
        self.mgr = Manager()
        self.labels = labels
        self.dim_reduc = None
        
    def onehot_to_labels(self, onehot, labels):
        return np.array([
            labels[np.nonzero(sample)[0][0]] 
            for sample in onehot
        ])
    
    def train_test_split(self, n_stim, trials_per_stim, test_size):
        """
        Build a custom train/test splitter. Gives an equal number of training 
        and test trial instances for each stimulus.
        """
        # save arguemnts as attributes
        # to be used by self.cross_validate
        self.n_stim = n_stim

        # instantiate shuffler
        rng = np.random.default_rng()
        
        # make sure the train/test split is even
        if trials_per_stim%(1/test_size) != 0:
            m = f"""
            The number of trials in the training set with the given
            test size ({test_size})is not a factor of the total 
            number of trials.
            """
            raise _NotAFactor(m)
        
        # get the number of train/test samples
        n_test = int(test_size*trials_per_stim)
        n_train = int((1-test_size)*trials_per_stim)

        ######## split Y ########
        
        # delete cache before starting
        try:
            del self.Y_train
        except AttributeError:
            pass
        try:
            del self.Y_test
        except AttributeError:
            pass
        
        iterY = copy.deepcopy(self.Y)
        if self.Ydim == 1:
            # reshape and shuffle
            iterY = iterY.reshape(n_stim, trials_per_stim)
            rng.shuffle(iterY, axis = 1)

            # get the train and test samples and revert to original shape
            self.Y_test = iterY[:,:n_test].reshape(n_stim*n_test)
            self.Y_train = iterY[:,n_test:].reshape(n_stim*n_train)
            del iterY
        elif self.Ydim > 2:
            raise _InvalidYDim(self.Ydim)
        else:
            cdim = iterY.shape[0]
            iterY = iterY.reshape(cdim, n_stim, trials_per_stim)
            rng.shuffle(iterY, axis = 2)

            # get the train and test samples and revert to original shape
            self.Y_test = iterY[:,:,:n_test].reshape(cdim, n_stim*n_test)
            self.Y_train = iterY[:,:,n_test:].reshape(cdim, n_stim*n_train)
            del iterY
        
        ######## split X ########
        
        # delete cache before starting
        try:
            del self.X_train
        except AttributeError:
            pass
        try:
            del self.X_test
        except AttributeError:
            pass
        
        n_test = int(test_size*trials_per_stim)
        n_train = int((1-test_size)*trials_per_stim)
        
        # make a copy of the data structure
        if type(self.X) == sparse._coo.core.COO:
            iterX = self.X.todense().astype(np.int16)
        else:
            iterX = copy.deepcopy(self.X).astype(np.int16)
        
        # reshape and shuffle
        iterX = iterX.reshape(
            iterX.shape[0],
            n_stim,
            trials_per_stim, 
            iterX.shape[-1]
        )
        rng = np.random.default_rng()
        rng.shuffle(iterX, axis = 2)
        
        # get the train and test samples and revert to original shape
        self.X_test = iterX[:,:,:n_test,:].reshape(
            iterX.shape[0],n_stim*n_test,iterX.shape[-1]
        )

        self.X_train = iterX[:,:,n_test:,:].reshape(
            iterX.shape[0],n_stim*n_train,iterX.shape[-1]
        )
        
        # delete the copy from memory
        del iterX
        
    def optimize_hyperparams(self, X, Y):
        
        methods = {
            'BayesSearchCV': BayesSearchCV(
                self.model,
                self.params['params'],
                cv = 3,
                scoring = 'accuracy',
                n_iter = 10,
                n_points  = 5
            ),
            'GridSearchCV': GridSearchCV(
                self.model,
                self.params['params'],
                cv = 3,
                scoring = 'accuracy'
            ),
            'RandomizedSearchCV': RandomizedSearchCV(
                self.model,
                self.params['params'],
                cv = 3,
                scoring = 'accuracy',
                n_iter = 10,
            )
            
        }
        
        hpfitter = methods[self.params['method']]
        hpfitter.fit(X, Y)
        self.hyperparams = hpfitter.best_params_
        for key, val, in self.hyperparams.items():
                self.model.__dict__[key] = val
                
    def feature_embed_time(self, X, t, lag):
        if t-lag<0:
            raise _LagTooLong(lag, t)
            
        X = X[:,:,t-lag:t+1]
        duration = X.shape[2]
        n_trials = X.shape[1]
        n_neurons = X.shape[0]
       
        X = X.transpose(0,2,1)
        return X.reshape(n_neurons*duration, n_trials)
    
    def sample_embed_time(self, X, t, lag):
        if t-lag<0:
            raise _LagTooLong(lag, t)
            
        X = X[:,:,t-lag:t+1]
        duration = X.shape[2]
        n_trials = X.shape[1]
        n_neurons = X.shape[0]
        
        return X.reshape(n_neurons, duration*n_trials)
    
    def reduce_dimensions(self, X, Y, method):
        if method == 'pls':
            pls = PLSRegression(10)
            return pls.fit_transform(X, Y)[0]
        if method == 'pca':
            pca = PCA()
            return pca.fit_transform(X)[:,:self.n_neurons]
    
    def fit(self, t = None, dim_reduc = None, **embedding_params):
        
        self.scaler = StandardScaler()
        if self.Ydim == 1:
            Y = self.Y_train
        elif self.Ydim > 2:
            raise _InvalidYDim(self.Ydim)
        else:
            Y = self.Y_train.T
            
        if t == None:
            raise _NoTimeSliceGiven("fit")
            
        if embedding_params:
            if 'space' not in embedding_params['embedding_params'].keys():
                raise _MissingEmbeddingArgs('space')
                
            if 'lag' not in embedding_params['embedding_params'].keys():
                raise _MissingEmbeddingArgs('lag')
                                      
            # extract the embedding parameters         
            self.embedding_params = embedding_params['embedding_params']
            self.space_selection = self.embedding_params['space']        
            lag = self.embedding_params['lag']
            
            if self.space_selection == 'feature':
                X = self.feature_embed_time(self.X_train, t, lag)
                X = self.scaler.fit_transform(X.T)
                
            elif self.space_selection == 'sample':
                X = self.sample_embed_time(self.X_train, t, lag)
                X = self.scaler.fit_transform(X.T)
                
                # extend Y training data to match
                if self.Ydim == 1:
                    Y = np.array([
                        np.array([i]*lag) for i in Y
                    ]).flatten()
                    
                elif self.Ydim == 2:
                    Y = np.array([
                        np.array([i]*(lag+1)) for i in Y
                    ]).reshape(X.shape[0], self.n_stim)                  
            else:
                raise _InvalidSpaceArg(space_selection)
        else:
            X = self.scaler.fit_transform(self.X_train[:,:,t].T) 

        if self.boost:
            self.model = AdaBoostClassifier(
                base_estimator=self.model, 
                random_state=42, 
                algorithm="SAMME", 
                n_estimators=5
            )
        else:
            pass

        #fitting with the optimal parameters
        
        if dim_reduc != None:
            self.dim_reduc = dim_reduc
            X = self.reduce_dimensions(X, Y, dim_reduc)
        self.model.fit(X, Y)  
        
    def score(self, t = None):
        
        if self.Ydim == 1:
            Y = self.Y_test
        elif self.Ydim > 2:
            raise _InvalidYDim(self.Ydim)
        else:
            Y = self.Y_test.T
            
        if t == None:
            raise _NoTimeSliceGiven("score")
            
        try:
            lag = self.embedding_params['lag']
            if self.space_selection == 'feature':
                X = self.feature_embed_time(self.X_test, t, lag)              
                X = self.scaler.fit_transform(X.T)
                
            elif self.space_selection == 'sample':
                X = self.sample_embed_time(self.X_test, t, lag)
                X = self.scaler.fit_transform(X.T)
                
                # extend Y test data to match
                if self.Ydim == 1:
                    Y = np.array([
                        np.array([i]*lag) for i in Y
                    ]).flatten()
                    
                elif self.Ydim == 2:
                    Y = np.array([
                        np.array([i]*(lag+1)) for i in Y
                    ]).reshape(X.shape[0], self.n_stim)  
                
            else:
                raise _InvalidSpaceArg(space_selection)
        except AttributeError:  
            X = self.scaler.transform(self.X_test[:,:,t].T)
                
        #evaluating accuracy
        if self.dim_reduc != None:
            X = self.reduce_dimensions(X, Y, self.dim_reduc)
        pred = self.model.predict(X)
        
        if self.Ydim == 1:
            score = (pred==Y).mean()
        if self.Ydim == 2:
            score = np.all(np.equal(pred, Y), axis=1).mean(0)
            
        if np.all(self.labels != None):
            truelabels = self.onehot_to_labels(Y, self.labels)
            predictedlabels = self.onehot_to_labels(pred, self.labels)
            cm = confusion_matrix(truelabels, predictedlabels)
        else:
            cm = confusion_matrix(Y, pred)
            
        return score, cm
    
    def _run_fold(
            self,
            Xtrain, Xtest,
            Ytrain, Ytest,
            models, scores, cms,
            optimize, *args
    ):
        if optimize:
            self.optimize_hyperparams(Xtrain, Ytrain)
        model = copy.deepcopy(self.model)
        model.fit(Xtrain, Ytrain)

        ## get the score and the confusion matrix
        pred = model.predict(Xtest)
        if self.Ydim == 1:
            score = (pred==Ytest).mean()
        if self.Ydim == 2:
            score = np.all(np.equal(pred, Ytest), axis=1).mean(0)

        if np.all(self.labels != None):
            truelabels = self.onehot_to_labels(Ytest, self.labels)
            predictedlabels = self.onehot_to_labels(pred, self.labels)
            cm = confusion_matrix(truelabels, predictedlabels)
        else:
            cm = confusion_matrix(Ytest, pred)

        ## append data to collection lists
        models.append(model)
        scores.append(score)
        cms.append(cm)

        # clear the fold data from memory
        del Xtrain;del Xtest
        del Ytrain;del Ytest
        del score;del cm
        
    def cross_validate(self, k, t, optimize = False, **embedding_params):
        
        trials_per_stim = int(self.X_train.shape[1]/self.n_stim)
        if trials_per_stim%k != 0:
            m = f"""
            The number of folds ({k}) is not a factor of the
            total number of trials per stimulus in the training set.
            """
            raise _NotAFactor(m)
        else:
            stim_per_fold = int(trials_per_stim/k)
        if t == None:
            raise _NoTimeSliceGiven()

        ## Reshape X_train
        X_cv = self.X_train.reshape(
            self.X_train.shape[0], 
            self.n_stim, 
            trials_per_stim, 
            self.X_train.shape[-1]
        )

        X_cv = X_cv.reshape(
            self.X_train.shape[0], # dim 0 = Neurons
            self.n_stim, # dim 1 = Stims
            k, # dim 2 = folds
            stim_per_fold, # dim 3 = stim trials per fold
            self.X_train.shape[-1] # dim 4 = time
        )

        ## Reshape Y_train
        if self.Ydim == 1:
            Y_cv = self.Y_train.reshape(self.n_stim, trials_per_stim)
            Y_cv = Y_cv.reshape(
                self.n_stim, # dim 0 = Stims
                k, # dim 1 = folds
                stim_per_fold # dim 2 = stim trials per fold
            )
        elif self.Ydim > 2:
            raise _InvalidYDim(self.Ydim)
        else:
            Y_cv = self.Y_train.reshape(
                self.Y_train.shape[0], 
                self.n_stim, 
                trials_per_stim
            )

            Y_cv = Y_cv.reshape(
                self.Y_train.shape[0], # number of classes
                self.n_stim, # dim 0 = Stims
                k, # dim 1 = folds
                stim_per_fold # dim 2 = stim trials per fold
            )    

        ## initialize lists to store the data for each fold
      
        models = self.mgr.list()
        scores = self.mgr.list()
        cms = self.mgr.list()
        
        # list to hold the processes
        processes = []
        
        ## iterate through the folds
        for i in range(X_cv.shape[2]):
            
            # create a mask to index all but the test set
            mask = np.ones(X_cv.shape[2], dtype = bool)
            mask[i] = 0

            # get the X test set
            Xtest = X_cv[:,:,i,:,:].reshape(
                X_cv.shape[0], # dim 0 = neurons
                X_cv.shape[1]*X_cv.shape[3], # dim 1 = stims x stim trials per fold
                X_cv.shape[-1] # dim 2 = time
            )
            
            # get the X train set
            Xtrain = X_cv[:,:,mask,:,:].reshape(
                X_cv.shape[0],
                X_cv.shape[1]*X_cv.shape[3]*(X_cv.shape[2]-1),
                X_cv.shape[-1]
            )
            
            if self.Ydim == 1:
                # get the Y test set
                Ytest = Y_cv[:,i,:].reshape(
                    Y_cv.shape[0]*Y_cv.shape[2]
                )

                # get the Y train set
                Ytrain = Y_cv[:,mask,:].reshape(
                    Y_cv.shape[0]*Y_cv.shape[2]*(Y_cv.shape[1]-1)
                )
                
            if self.Ydim == 2:
                # get the Y test set
                Ytest = Y_cv[:,:,i,:].reshape(
                    Y_cv.shape[0],
                    Y_cv.shape[1]*Y_cv.shape[3]
                ).T
                
                # get the Y train set
                Ytrain = Y_cv[:,:,mask,:].reshape(
                    Y_cv.shape[0],
                    Y_cv.shape[1]*Y_cv.shape[3]*(Y_cv.shape[2]-1)
                ).T      

            ## fit the model
            self.scaler = StandardScaler()
            
            if t == None:
                raise _NoTimeSliceGiven("fit")

            if embedding_params:
                if 'space' not in embedding_params['embedding_params'].keys():
                    raise _MissingEmbeddingArgs('space')

                if 'lag' not in embedding_params['embedding_params'].keys():
                    raise _MissingEmbeddingArgs('lag')

                # extract the embedding parameters         
                self.embedding_params = embedding_params['embedding_params']
                self.space_selection = self.embedding_params['space']        
                lag = self.embedding_params['lag']

                if self.space_selection == 'feature':
                    
                    # prepare Xtrain
                    Xtrain = self.feature_embed_time(Xtrain, t, lag)
                    Xtrain = self.scaler.fit_transform(Xtrain.T)
                    
                    # prepare Xtest
                    Xtest = self.feature_embed_time(Xtest, t, lag)
                    Xtest = self.scaler.fit_transform(Xtest.T)
                    processes.append(
                        Process(
                        target = self._run_fold,
                        args = (
                                Xtrain, Xtest, 
                                Ytrain, Ytest, 
                                models, scores, cms,
                                optimize, lag
                            )
                        )
                    )
                elif self.space_selection == 'sample':
                    
                    # prepare Xtrain
                    Xtrain = self.sample_embed_time(Xtrain, t, lag)
                    Xtrain = self.scaler.fit_transform(Xtrain.T)
                    
                    # prepare Xtest
                    Xtest = self.sample_embed_time(Xtest, t, lag)
                    Xtest = self.scaler.fit_transform(Xtest.T)
                    
                    # extend Y training data to match
                    if self.Ydim == 1:
                        Ytrain = np.array([
                            np.array([i]*lag) for i in Ytrain
                        ]).flatten()
                        
                    elif self.Ydim == 2:
                        Ytrain = np.array([
                            np.array([i]*(lag+1)) for i in Ytrain
                        ]).reshape(Xtrain.shape[0], self.n_stim)   
                        
                    # extend Y test data to match
                    if self.Ydim == 1:
                        Ytest = np.array([
                            np.array([i]*lag) for i in Ytest
                        ]).flatten()
                        
                    elif self.Ydim == 2:
                        Ytest = np.array([
                            np.array([i]*(lag+1)) for i in Ytest
                        ]).reshape(Xtest.shape[0], self.n_stim)  
                        
                    processes.append(
                        Process(
                        target = self._run_fold,
                        args = (
                                Xtrain, Xtest, 
                                Ytrain, Ytest, 
                                models, scores, cms,
                                optimize, lag
                            )
                        )
                    )
                    
                else:
                    raise _InvalidSpaceArg(space_selection)
            else:
                Xtrain = self.scaler.fit_transform(Xtrain[:,:,t].T) 
                Xtest = self.scaler.transform(Xtest[:,:,t].T)               
        
                processes.append(
                    Process(
                    target = self._run_fold,
                    args = (
                            Xtrain, Xtest, 
                            Ytrain, Ytest, 
                            models, scores, cms,
                            optimize
                        )
                    )
                )
                
            # clear the fold data from memory
            del Xtrain;del Xtest
            del Ytrain;del Ytest
            
        ## start and join the processes
        [p.start() for p in processes]
        [p.join() for p in processes]

        del X_cv;del Y_cv
        return list(models), list(scores), list(cms)
    
    def clear_cache(self):
        self.__dict__ = {}
        
class _InvalidYDim(Exception):
    """
    Exception raised for when the dimension of 
    the dependent variables Y exceeds 2.
    """

    def __init__(self, arg):
        self.message = f"""
        The dependent variables Y must be at most a two-dimensional matrix. 
        Got a dimension of {arg}
        """
        super().__init__(self.message)
        
class _MissingEmbeddingArgs(Exception):
    """
    Exception raised for when feature or sample embedding 
    is used but necessary arguemnts are missing.
    """

    def __init__(self, arg):
        self.message = f"""
        t = User tried to use time embedding, but no {arg} parameter was given.
        """
        super().__init__(self.message)
        
        
class _LagTooLong(Exception):
    """
    Exception raised for when the lag for time embedding
    is longer than the time point being predicted.
    """

    def __init__(self, arg1, arg2):
        self.message = f"""
        t = Lag of length ({arg1}) is longer than the timepoint being predicted ({arg2}).
        """
        super().__init__(self.message)
        
class _InvalidSpaceArg(Exception):         
    """
    Exception raised for when the space arguemt for time 
    embedding is invalid.
    """

    def __init__(self, arg):
        self.message = f"""
        t = '{arg}' is an invalid argument for the space parameter.
        """
        super().__init__(self.message)
        
class _NotAFactor(Exception):
    """
    Exception raised for when trial subsets are 
    not factors of the total number of trials.
    """

    def __init__(self, message):
        super().__init__(message)
        
class _NoTimeSliceGiven(Exception):
    """
    Exception raised for when no time slice is
    given in the fit, score, and cross_validate
    methods.
    """

    def __init__(self, arg):
        self.message = f"""
        No time slice was specified for decoding. Pass a timepoint 
        argument or pass '{arg}_mean = True' to {arg} the time averaged
        trials.
        """
        super().__init__(self.message)