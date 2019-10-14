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

    def count(self, name = 'e1' , meta = 'count', nodes = None, debug = False):
        """
        count number of points in the neighborhood
        """
        self.estimates[name] = {}
        self.estimates[name]['vname'] = None
        self.estimates[name][meta] = meta

        if nodes is None:
            nodes = self.nodes


        def f(i):
            # update data selected around target point
            self.search.update([self.x0[i],self.y0[i],self.z0[i]])
            if debug:
                return np.sum(self.search.test), self.search.row_id[self.search.test]
            else:
                return np.sum(self.search.test), None


        # apply the estimator to each target
        self.estimates[name]['estimate'] = np.array(list(map(f,nodes)))
        
        
    def id_power(self, name = 'e1' , meta = 'id^p estimate', nodes = None, power = 2, isotropic = True, debug = False):
        """
        ID power estimate
        """
        
        self.estimates[name] = {}
        self.estimates[name]['vname'] = list(self.search.v.keys())
        self.estimates[name]['meta'] = meta

        if nodes is None:
            nodes = self.nodes


        def f(i):
            # update data selected around target point
            self.search.update([self.x0[i],self.y0[i],self.z0[i]])
            
            # get distance
            if isotropic:
                d = self.search.get_distances(0)
            else:
                d = self.search.get_distances(0)**power

            # estimate for each variable
            r = []
            for k in self.search.v.keys():
                r.append(np.sum(self.search.v[k][self.search.test]*d,axis = 0)/np.sum(d))
            
            if debug: 
                return r + [d] + [self.search.row_id[self.search.test]]
            else:
                return r + [None]


        # apply the estimator to each target
        self.estimates[name]['estimate'] = np.array([f(i) for i in nodes])
        
