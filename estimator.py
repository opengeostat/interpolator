import numpy as np


class Estimator():
    """
    TODO:
    """

    def __init__(self, x0,y0,z0, search):

        # set data
        assert np.array(x0).ndim == 1
        assert np.array(y0).ndim == 1
        assert np.array(z0).ndim == 1
        assert np.array(x0).shape==np.array(y0).shape==np.array(z0).shape
        self.x0 = np.array(x0)
        self.y0 = np.array(y0)
        self.z0 = np.array(z0)
        
        self.n0 = self.x0.shape[0]
        self.nodes = np.arange(self.n0)
        
        #set search (a class Neighborhood instance containing search data and parameters)
        self.search = search
        
        # we save results in a dictionary
        self.estimates = {}
        
    def mean(self, name = 'e1' , meta = 'count', target = None, debug = True):
        """
        count number of points
        """
        self.estimates[name] = {}
        self.estimates[name]['vname'] = None
        self.estimates[name][meta] = meta
        
        if nodes is None:
            nodes = self.nodes


        def f(i):
            self.search.update([self.x0[i],self.y0[i],self.z0[i]])
            if debug:
                pass
            else:
                return np.sum(self.search.test)
            
        
        # apply the estimator to each target
        self.estimates[name]['estimate'] = np.array(list(map(f,nodes)))
