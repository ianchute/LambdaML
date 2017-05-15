import numpy as np
from collections import Iterable

class LambdaUtils:

    @staticmethod
    def _prime(f, x, h=np.finfo(np.float16).eps):
        """Numerically computes the first derivative of f at point x."""
        return (f(x + h) - f(x - h)) / h * 2
    
    @staticmethod
    def dependent(f,p,key,X,Y):
        """Creates a function that is dependent on the model parameter p[key]"""
        def dependent_f(z):
            _p = p.copy()
            _p[key] = z
            return f(_p,X,Y)
        return dependent_f

    @staticmethod
    def dependent_a(f,p,key,i,X,Y):
        """Creates a function that is dependent on the model parameter p[key][i]"""
        def dependent_f(z):
            _p = p.copy()
            _p[key] = _p[key].copy()
            _p[key][i] = z
            return f(_p,X,Y)
        return dependent_f

    @staticmethod
    def prime(f, p, key, X, Y):
        """Numerically computes the first derivative of f with respect to p[key]"""
        if isinstance(p[key], Iterable):
            return np.array(list(map(
                lambda ix: LambdaUtils._prime(LambdaUtils.dependent_a(f,p,key,ix[0],X,Y), p[key][ix[0]]), 
                enumerate(p[key]))))
        else:
            return LambdaUtils._prime(LambdaUtils.dependent(f,p,key,X,Y), p[key])
    
    @staticmethod
    def log_likelihood(f):
        """Generates the log likelihood function of f"""
        def f_likelihood(p,X,Y):
            likelihood = 0
            for x,y in zip(X,Y):
                _f = f(x,p)
                if _f > 0 and _f < 1:
                    likelihood += y * np.log(_f) + (1 - y) * np.log(1 - _f)
            return likelihood - LambdaUtils.l1_regularization(p) - LambdaUtils.l2_regularization(p)
        return f_likelihood

    @staticmethod
    def l1_regularization(p):
        total = 0
        for v in p.values():
            if isinstance(v, Iterable):
                total += v.sum()
            else:
                total += v
        return total

    @staticmethod
    def l2_regularization(p):
        total = 0
        for v in p.values():
            if isinstance(v, Iterable):
                total += np.abs(v).sum()
            else:
                total += np.square(v)
        return total
