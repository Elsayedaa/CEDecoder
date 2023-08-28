## Version 5
## Changes:

# the optimize_hyperparams method is now an independent method 
# removed the fit_timepoints_as_samples and score_timepoints_as_samples method
# the fit_repeats_as_samples and score_repeats_as_samples methods have been reneamed to just "fit" and "score"
# added the cross_validate method for k fold cross validation & enabled multiprocessing for cross validation

import copy
import numpy as np
import sparse
from multiprocessing import Process, Manager
from sklearn.preprocessing import scale
from skopt import BayesSearchCV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix

class decoder:
    def __init__(self, X, Y, model: str, params = None, boost = False):
        """
        X: Must be a 3D matrix of shape (N(Neurons), N(Trials), N(Timepoints))
        Y: Must be a 1D array of categorical labels corresponding to the stimulus shown at each trial. 
        model: Must be an instance of a sklearn classification model
        params: Must be a dictionary of hyperparameters to be optimized
        boost: Boolean, default = False
        """
        self.X = X
        self.Y = Y
        self.model = model
        self.params = params
        self.boost = boost
        self.mgr = Manager()

    def train_test_split(self, n_stim, trials_per_stim, test_size, skip_Y = False):
        """
        Build a custom train/test splitter. Gives an equal number of training 
        and test trial instances for each stimulus.
        """
        # save arguemnts as attributes
        # to be used by self.cross_validate
        self.n_stim = n_stim
        
        # to be used by self.optimize_hyperparams
        self.skip_Y = skip_Y
        
        # instantiate shuffler
        rng = np.random.default_rng()
        
        # make sure the train/test split is even
        if trials_per_stim%(1/test_size) != 0:
            return "Specified test size does not form an even split."
        
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
        
        if skip_Y == False:
            # make a copy of the data structure
            iterY = copy.deepcopy(self.Y)

            # reshape and shuffle
            iterY = iterY.reshape(n_stim, trials_per_stim)
            rng.shuffle(iterY, axis = 1)

            # get the train and test samples and revert to original shape
            self.Y_test = iterY[:,:n_test].reshape(n_stim*n_test)
            self.Y_train = iterY[:,n_test:].reshape(n_stim*n_train)

            # delete the copy from memory
            del iterY
        else:
            pass
        
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
        
    def fit(self, fit_mean = False, t = None):
        
        Y = self.Y_train
        
        if fit_mean:
            X = scale(self.X_train.mean(2).T)
        else:
            if t == None:
                return "A time slice must be given."
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
        
        
    def score(self, score_mean = False, t = None):
        
        Y = self.Y_test

        if score_mean:
            X = scale(self.X_test.mean(2).T)
        else:
            if t == None:
                return "A time slice must be given."
            else:
                X = scale(self.X_test[:,:,t].T)
                
        #evaluating accuracy
        pred = self.model.predict(X)
        score = (pred==Y).mean()
        cm = confusion_matrix(Y, pred)
        return score, cm
    
    def _run_fold(
            self,
            Xtrain, Xtest,
            Ytrain, Ytest,
            models, scores, cms,
            optimize
    ):
        if optimize:
            self.optimize_hyperparams(Xtrain, Ytrain)
        model = copy.deepcopy(self.model)
        model.fit(Xtrain, Ytrain)

        ## get the score and the confusion matrix
        pred = model.predict(Xtest)
        score = (pred==Ytest).mean()
        cm = confusion_matrix(Ytest, pred)

        ## append data to collection lists
        models.append(model)
        scores.append(score)
        cms.append(cm)

        # clear the fold data from memory
        del Xtrain;del Xtest
        del Ytrain;del Ytest
        del score;del cm
        
    def cross_validate(self, k, t, optimize = False):
        trials_per_stim = int(self.X_train.shape[1]/self.n_stim)
        if trials_per_stim%k != 0:
            return "Number of folds does not divide evenly into the number of trials per stimulus."
        else:
            stim_per_fold = int(trials_per_stim/k)

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
        Y_cv = self.Y_train.reshape(self.n_stim, trials_per_stim)
        Y_cv = Y_cv.reshape(
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

            # get the Y test set
            Ytest = Y_cv[:,i,:].reshape(
                Y_cv.shape[0]*Y_cv.shape[2]
            )

            # get the X train set
            Xtrain = X_cv[:,:,mask,:,:].reshape(
                X_cv.shape[0],
                X_cv.shape[1]*X_cv.shape[3]*(X_cv.shape[2]-1),
                X_cv.shape[-1]
            )

            # get the Y train set
            Ytrain = Y_cv[:,mask,:].reshape(
                Y_cv.shape[0]*Y_cv.shape[2]*(Y_cv.shape[1]-1)
            )

            ## fit the model
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