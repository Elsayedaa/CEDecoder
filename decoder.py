## Version 3
## Changes:

# Option to use BayesSearchCV and RandomizedSearchCV hyperparameter optimization

# Added new models: KNN, Gaussian Process, Naive Bayes

# Added Ada boost ensemble

# Made some memory utilization optimizations.
import copy
import numpy as np
import sparse
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from skopt import BayesSearchCV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix

class decoder:
    def __init__(self, X, Y, model: str, params = None, boost = False):
        """
        X: Must be a 3D matrix of shape (N(Neurons), N(Trials), N(Timepoints))
        Y: Must be a 1D array of categorical labels corresponding to the stimulus
        Shown at each trial. 
        """
        self.X = X
        self.Y = Y
        
        self.model_selection = {
            'SVM': SVC(random_state = 42),
            'KNN': KNeighborsClassifier(),
            'GPC': GaussianProcessClassifier(random_state = 42),
            'NBC': GaussianNB()
        }
        
        self.model = self.model_selection[model]
        self.params = params
        self.boost = boost

    def train_test_split(self, n_stim, trials_per_stim, test_size, skip_Y = False):
        """
        Build a custom train/test splitter. Gives an equal number of training 
        and test trial instances for each stimulus.
        """
        # save arguemnts as attributes
        # to be used by self.fit_timepoints_as_samples
        self.n_stim = n_stim
        self.trials_per_stim = trials_per_stim
        self.test_size = test_size
        
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
            iterX = self.X.todense()
        else:
            iterX = copy.deepcopy(self.X)
        
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
        
    def optimize_hyperparams(self, X, Y, params, method: str):
        
        methods = {
            'BayesSearchCV': BayesSearchCV(
                self.model,
                params,
                cv = 3,
                scoring = 'accuracy',
                n_jobs = 4,
                n_iter = 10,
                n_points  = 5
            ),
            'GridSearchCV': GridSearchCV(
                self.model,
                params,
                cv = 3,
                scoring = 'accuracy',
                n_jobs = 4
            ),
            'RandomizedSearchCV': RandomizedSearchCV(
                self.model,
                params,
                cv = 3,
                scoring = 'accuracy',
                n_jobs = 4,
                n_iter = 10,
            )
            
        }
        
        hpfitter = methods[method]
        hpfitter.fit(X, Y)
        self.hyperparams = hpfitter.best_params_
        
    def fit_repeats_as_samples(self, fit_mean = False, t = None):
        
        Y = self.Y_train
        
        if fit_mean:
            X = scale(self.X_train.mean(2).T)
        else:
            if t == None:
                return "A time slice must be given."
            else:
                X = scale(self.X_train[:,:,t].T)
                
        if self.params != None:
            self.optimize_hyperparams(X, Y, self.params['params'], self.params['method'])
            for key, val, in self.hyperparams.items():
                self.model.__dict__[key] = val
        else:
            pass
        
        if self.boost:
            self.model = AdaBoostClassifier(
                estimator=self.model, 
                random_state=42, 
                algorithm="SAMME", 
                n_estimators=5
            )
        else:
            pass

        #fitting with the optimal parameters
        self.model.fit(X, Y)   
        
    def fit_timepoints_as_samples(self, window: slice):
        
        Y = self.Y
        
        X = self.X_train
        
        X = X.reshape(
            X.shape[0], 
            self.n_stim, 
            int(self.trials_per_stim*(1-self.test_size)), 
            X.shape[-1]
        ).mean(2)
        
        X = X[:,:,window].reshape(
            X.shape[0], 
            int((window.stop-window.start)*self.n_stim)
        )
        X = scale(X.T)
        
        if self.params != None:
            self.optimize_hyperparams(X, Y, self.params['params'], self.params['method'])
            for key, val, in self.hyperparams.items():
                self.model.__dict__[key] = val
        else:
            pass
        
        if self.boost:
            self.model = AdaBoostClassifier(
                estimator=self.model, 
                random_state=42, 
                algorithm="SAMME", 
                n_estimators=5
            )
        else:
            pass

        #fitting with the optimal parameters
        self.model.fit(X, Y)   
        
    def score_repeats_as_samples(self, score_mean = False, t = None):
        
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
    
    def score_timepoints_as_samples(self, window: slice):
        
        Y = self.Y
        
        X = self.X_test
        
        X = X.reshape(
            X.shape[0], 
            self.n_stim, 
            int(self.trials_per_stim*(1-self.test_size)), 
            X.shape[-1]
        ).mean(2)
            
        X = X[:,:,window].reshape(
            X.shape[0], 
            int((window.stop-window.start)*self.n_stim)
        )
        X = scale(X.T) 
        
        #evaluating accuracy
        pred = self.model.predict(X)
        score = (pred==Y).mean()
        cm = confusion_matrix(Y, pred)
        return score, cm
    
    def clear_cache(self):
        self.__dict__ = {}