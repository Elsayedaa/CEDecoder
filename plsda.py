from sklearn.cross_decomposition import PLSRegression
import numpy as np

def create_supermodel(models, param_keys):
    supermodel = models[0]
    for key in param_keys:
        supermodel.__dict__[key] = [supermodel.__dict__[key]]
    for model in models[1:]:
        for key in param_keys:
            supermodel.__dict__[key].append(model.__dict__[key])
    for key in param_keys:
        supermodel.__dict__[key] = np.array(supermodel.__dict__[key]).mean(0)
        
    return supermodel

class PLSDA(PLSRegression):
    def __init__(
        self,
        n_components = 2,
        scale = True,
        max_iter = 500,
        tol = 1e-06,
        copy = True
    ):
        super().__init__(
            n_components = n_components,
            scale = scale,
            max_iter = max_iter,
            tol = tol,
            copy = copy
        )
    
    def reg_predict(self, X):
        return super().predict(X)
        
    def predict(self, X):
        Y_prob = super().predict(X)
        Y_pred = np.zeros(Y_prob.shape)
        for i in range(Y_prob.shape[0]):
            # find the index with the highest value in each row
            pred_i = np.where(
                Y_prob[i] == Y_prob[i].max()
            )[0][0] 
            # assign 1 to the index with the highest value
            Y_pred[i][pred_i] = 1
        return Y_pred
    
    def score(self, X, Y):
        pred = self.predict(X)
        score = np.all(np.equal(pred, Y.T), axis=1).mean(0)
        return score