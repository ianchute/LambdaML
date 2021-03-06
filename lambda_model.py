from lambda_utils import LambdaUtils

class LambdaClassifierModel:
    """
    A generic classifier model that accepts a probability function f and a parameter set p.
    f must have results within [0,1] which reflects the probability of being in the positive class.
    p must be a dictionary containing the parameters of the model.

    Calling fit will automatically fit all parameters via Maximum Likelihood Estimation.
    Calling predict will return class labels.
    """
    def __init__(self, f, p, l1_factor=0.05, l2_factor=0.01):
        """
        A generic classifier model that accepts a probability function f and a parameter set p.
        f must have results within [0,1] which reflects the probability of being in a class.
        p must be a dictionary containing the parameters of the model.
        """
        self.f = f
        self.p = p
        self.likelihood = LambdaUtils.log_likelihood(self.f, l1_factor, l2_factor)

    def fit(self,X,Y,n_iter=10,alpha=0.01):
        """
        Automatically fit all parameters via Maximum Likelihood Estimation
        """
        p_keys = self.p.keys()
        updates = {}
        for i in range(n_iter):

            # Acquire updates.
            for key in p_keys:
                updates[key] = alpha * LambdaUtils.prime(self.likelihood,self.p,key,X,Y)
            
            # Apply updates.
            for key in p_keys:
                self.p[key] += updates[key]
    
    def compute_log_likelihood(self,X,Y):
        """
        Returns the log likelihood of the model on X producing Y.
        """
        return self.likelihood(self.p,X,Y)
    
    def predict(self,X):
        """
        Predict will return class labels.
        """
        return list(map(lambda x: 0 if self.f(x,self.p) < 0.5 else 1, X))

    def predict_probability(self,X):
        """
        Predict will return probabilities of belonging to the positive class.
        """
        return list(map(lambda x: self.f(x,self.p), X))