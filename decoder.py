## Version 6
## Changes:

# time invariant model option now available if
# the argument t = 'inviarant' is passed into the
# following methods:
#    - self.fit()
#    - self.score()
#    - self.cross_validate()

# time invariant delay embedded model option now available if
# the argument t = 'embedded' is passed into the
# following methods:
#    - self.fit()
#    - self.score()
#    - self.cross_validate()
# for the delay embedded model, an additional keyword argument
# 'embedding_params' must be passed. The embedding_params argument
# should have the structure: {'time_lag': int, 'time_range': tuple}


import copy
import numpy as np
import sparse
from multiprocessing import Process, Manager
from sklearn.preprocessing import scale
from skopt import BayesSearchCV
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
        self.Ydim = len(Y.shape)
        self.model = model
        self.params = params
        self.boost = boost
        self.mgr = Manager()
        self.labels = labels
        
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
                
    def delay_embed(self, X, time_lag, time_range, duration):
        return np.array(
            [
                np.array(
                    [
                        X[x,:,t:t+time_lag].T 
                        for x in range(X.shape[0])
                    ]
                ).reshape(X.shape[0]*time_lag, X.shape[1]).T 
                for t in range(time_range[0], time_range[1])
            ], 
            dtype = np.int16
        ).T.reshape(X.shape[0]*time_lag, X.shape[1]*duration)
    
    def fit(self, t = None, **embedding_params):
        
        if self.Ydim == 1:
            Y = self.Y_train
        elif self.Ydim > 2:
            raise _InvalidYDim(self.Ydim)
        else:
            Y = self.Y_train.T
        
        if t == 'invariant':
            X = self.X_train
            duration = X.shape[2]
            n_trials = X.shape[1]
            X = X.reshape(X.shape[0], duration*n_trials)
            X = scale(X.T)
            Y = np.array([np.array([i]*duration) for i in Y]).reshape(n_trials*duration,self.n_stim)
        elif t == 'embedded':
            must_inclued_kwargs = ['time_lag', 'time_range']
            for kwarg in must_inclued_kwargs:
                if kwarg not in embedding_params['embedding_params'].keys():
                    raise _NoDelayEmbeddingArgs(kwarg)
                    
            # extract the embedding parameters        
            time_lag = embedding_params['embedding_params']['time_lag']
            time_range = embedding_params['embedding_params']['time_range']
            duration = len(range(time_range[0], time_range[1]))
            
            # delay embed the X data
            X = self.X_train
            n_trials = X.shape[1]
            X = scale(self.delay_embed(X, time_lag, time_range, duration).T)
            self.X_test = self.delay_embed(self.X_test, time_lag, time_range, duration).reshape(
                self.X_test.shape[0]*time_lag, self.X_test.shape[1], duration
            )
            
            # extend the Y data to match
            Y = np.array([np.array([i]*duration) for i in Y]).reshape(n_trials*duration,self.n_stim)
        else:
            if t == None:
                raise _NoTimeSliceGiven("fit")
            else:
                X = scale(self.X_train[:,:,t].T)
        
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
        self.model.fit(X, Y)   
        
    def score(self, t = None):
        
        if self.Ydim == 1:
            Y = self.Y_test
        elif self.Ydim > 2:
            raise _InvalidYDim(self.Ydim)
        else:
            Y = self.Y_test.T

        if t == 'invariant':
            X = self.X_test
            duration = X.shape[2]
            n_trials = X.shape[1]
            X = X.reshape(X.shape[0], duration*n_trials)
            X = scale(X.T)
            Y = np.array([np.array([i]*duration) for i in Y]).reshape(n_trials*duration,self.n_stim)
        elif t == 'embedded':
            X = self.X_test
            duration = X.shape[2]
            n_trials = X.shape[1]
            X = scale(self.X_test.reshape(
                self.X_test.shape[0],
                self.X_test.shape[1]*self.X_test.shape[2]
            ).T)
            Y = np.array([np.array([i]*duration) for i in Y]).reshape(n_trials*duration,self.n_stim)
        else:
            if t == None:
                raise _NoTimeSliceGiven("score")
            else:
                X = scale(self.X_test[:,:,t].T)
                
        #evaluating accuracy
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
            optimize, invariant_model = False,
            *args
    ):
        if optimize:
            self.optimize_hyperparams(Xtrain, Ytrain)
        model = copy.deepcopy(self.model)
        model.fit(Xtrain, Ytrain)

        ## get the score and the confusion matrix
        if invariant_model == True:
            duration = args[0]
            score = np.zeros(duration)
            cm = np.zeros((duration, self.n_stim, self.n_stim))
            for t in range(duration):
                xtest = scale(Xtest[:,:,t].T)
                pred = model.predict(xtest)
                if self.Ydim == 1:
                    score[t] = (pred==Ytest).mean()
                if self.Ydim == 2:
                    score[t] = np.all(np.equal(pred, Ytest), axis=1).mean(0)
                    
                if np.all(self.labels != None):
                    truelabels = self.onehot_to_labels(Ytest, self.labels)
                    predictedlabels = self.onehot_to_labels(pred, self.labels)
                    cm[t] = confusion_matrix(truelabels, predictedlabels)
                else:
                    cm[t] = confusion_matrix(Ytest, pred)
        else:
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
        print(X_cv.shape[2])
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
            if t == 'invariant':
                duration = Xtrain.shape[2]
                n_trials = Xtrain.shape[1]
                Xtrain = Xtrain.reshape(Xtrain.shape[0], duration*n_trials)
                Xtrain = scale(Xtrain.T)
                Ytrain = np.array([np.array([i]*duration) for i in Ytrain]).reshape(n_trials*duration,self.n_stim)
                processes.append(
                    Process(
                    target = self._run_fold,
                    args = (
                            Xtrain, Xtest, 
                            Ytrain, Ytest, 
                            models, scores, cms,
                            optimize, True, duration
                        )
                    )
                )
            elif t == 'embedded':
                must_inclued_kwargs = ['time_lag', 'time_range']
                for kwarg in must_inclued_kwargs:
                    if kwarg not in embedding_params['embedding_params'].keys():
                        raise _NoDelayEmbeddingArgs(kwarg)

                # extract the embedding parameters        
                time_lag = embedding_params['embedding_params']['time_lag']
                time_range = embedding_params['embedding_params']['time_range']
                duration = len(range(time_range[0], time_range[1]))

                # delay embed the X data
                n_trials = Xtrain.shape[1]
                Xtrain = scale(self.delay_embed(Xtrain, time_lag, time_range, duration).T)
                Xtest = self.delay_embed(Xtest, time_lag, time_range, duration).reshape(
                    Xtest.shape[0]*time_lag, Xtest.shape[1], duration
                )

                # extend the Y data to match
                Ytrain = np.array([np.array([i]*duration) for i in Ytrain]).reshape(n_trials*duration,self.n_stim)
                processes.append(
                    Process(
                    target = self._run_fold,
                    args = (
                            Xtrain, Xtest, 
                            Ytrain, Ytest, 
                            models, scores, cms,
                            optimize, True, duration
                        )
                    )
                )
            else:
                processes.append(
                    Process(
                    target = self._run_fold,
                    args = (
                            scale(Xtrain[:,:,t].T), scale(Xtest[:,:,t].T), 
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
        
class _NoDelayEmbeddingArgs(Exception):
    """
    Exception raised for when t = 'embedded' but no time_lag and/or
    time_range arguemnts are given.
    """

    def __init__(self, arg):
        self.message = f"""
        t = 'embedded' argument passed but no {arg} embedding parameter was given.
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