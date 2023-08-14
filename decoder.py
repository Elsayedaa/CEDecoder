## Version 1

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
    def train_test_split(self, n_stim, trials_per_stim, x_trial_axis, test_size):
        # save arguemnts as attributes
        # to be used by self.fit_timepoints_as_samples
        self.n_stim = n_stim
        self.trials_per_stim = trials_per_stim
        self.test_size = test_size
        
        self.stim_indices = np.array(list(
            zip(
                list(range(0, n_stim*trials_per_stim, trials_per_stim)),
                list(range(trials_per_stim, (n_stim*trials_per_stim)+trials_per_stim, trials_per_stim))
            )
        ))
        Y_train = []
        Y_test = []
        X_train = []
        X_test = []
        for i0, i1 in self.stim_indices:

            ## split X

            # shuffle trials
            rng = np.random.default_rng()
            X_stim = self.X[:,i0:i1,:]
            rng = np.random.default_rng()
            rng.shuffle(X_stim, axis = x_trial_axis)

            # generate slices
            x_train_slice = [slice(0,x) for x in list(self.X.shape)]
            x_train_slice[x_trial_axis] = slice(0, int(trials_per_stim*(1-test_size)))
            x_train_slice = tuple(x_train_slice)

            x_test_slice = [slice(0,x) for x in list(self.X.shape)]
            x_test_slice[x_trial_axis] = slice(int(trials_per_stim*(1-test_size)), trials_per_stim)
            x_test_slice = tuple(x_test_slice)

            # get the slice
            X_stim_train = X_stim[x_train_slice]
            X_stim_test = X_stim[x_test_slice]
            
            # append to train set and test set
            X_train.append(X_stim_train)
            X_test.append(X_stim_test)

            ## split Y
            
            # shuffle trials
            Y_stim = self.Y[i0:i1]
            random.shuffle(Y_stim)

            # get the slice
            Y_stim_train = Y_stim[:int(trials_per_stim*(1-test_size))]
            Y_stim_test = Y_stim[int(trials_per_stim*(1-test_size)):]

            # append to train set and test set
            Y_train += list(Y_stim_train)
            Y_test += list(Y_stim_test)

        # concatenate along the trial axis
        self.X_train = np.concatenate(X_train, axis = x_trial_axis)
        self.X_test = np.concatenate(X_test, axis = x_trial_axis)
        
        self.Y_train = Y_train
        self.Y_test = Y_test
    
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
            scoring = 'accuracy'
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
        
    def fit_timepoints_as_samples(self, window: slice):
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
    
    def score_timepoints_as_samples(self, window: slice):
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