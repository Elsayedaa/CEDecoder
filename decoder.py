## Version 2
## Changes:
# Shuffles were done inplace on the X & Y data structures in version 1,
# now they are done on deep copies.

# Made some memory utilization optimizations.
class decoder:
    def __init__(self, X, Y):
        """
        X: Must be a 3D matrix of shape (N(Neurons), N(Trials), N(Timepoints))
        Y: Must be a 1D array of categorical labels corresponding to the stimulus
        Shown at each trial. 
        """
        self.X = X
        self.Y = Y
        
    ## build a custom train/test splitter
    # gives an equal number of training and test
    # trial instances for each stimulus
    
    def apply_temporal_smoothing(self, trial_data, window, t_axis):
        return np.apply_along_axis(
                        lambda m: np.convolve(
                            m, np.ones(window)/window, 
                            mode='same'
                        ), axis=t_axis, arr=trial_data
                    )
    
    def train_test_split(self, n_stim, trials_per_stim, test_size, skip_Y = False):
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
        
    def optimize_hyperparams(self, X, Y):
        #using GridSearchCV to find the optimal values for C and gamma
        par = {
            'C':[x+1 for x in range(10)],
            'gamma':['scale', 'auto', 0.1, 1, 10],
            'kernel':['rbf']
        }

        hpfitter = GridSearchCV(
            SVC(),
            par,
            cv=5,
            scoring = 'accuracy',
            n_jobs = 8
        )

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
                
        self.optimize_hyperparams(X, Y)
        self.model = SVC(
            C=self.hyperparams['C'], 
            gamma = self.hyperparams['gamma'], 
            kernel = self.hyperparams['kernel'], 
            random_state = 42
        )

        #fitting with the optimal parameters
        self.model.fit(X, Y)   
        
    def fit_timepoints_as_samples(self, window: slice, smoothing = None):
        Y = self.Y
        X = self.X_train
        X = X.reshape(
            X.shape[0], 
            self.n_stim, 
            int(self.trials_per_stim*(1-self.test_size)), 
            X.shape[-1]
        ).mean(2)
        
        if smoothing == None:
            pass
        else:
            X = self.apply_temporal_smoothing(X, smoothing, 2)
            
        X = X[:,:,window].reshape(
            X.shape[0], 
            int((window.stop-window.start)*self.n_stim)
        )
        X = scale(X.T)
        self.optimize_hyperparams(X, Y)
        self.model = SVC(
            C=self.hyperparams['C'], 
            gamma = self.hyperparams['gamma'], 
            kernel = self.hyperparams['kernel'], 
            random_state = 42
        )

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
    
    def score_timepoints_as_samples(self, window: slice, smoothing = None):
        Y = self.Y
        X = self.X_test
        X = X.reshape(
            X.shape[0], 
            self.n_stim, 
            int(self.trials_per_stim*(1-self.test_size)), 
            X.shape[-1]
        ).mean(2)
        
        if smoothing == None:
            pass
        else:
            X = self.apply_temporal_smoothing(X, smoothing, 2)
            
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