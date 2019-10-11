import numpy as np
import concurrent.futures

DEG2RAD=3.141592654/180.0

class Neighborhood():
    """
    TODO:
    """

    def __init__(self,x,y,z, a= None, pivot = None, rot=None,test=None):

        # set data
        assert np.array(x).ndim == 1
        assert np.array(y).ndim == 1
        assert np.array(z).ndim == 1
        assert np.array(x).shape==np.array(y).shape==np.array(z).shape
        self.x = np.array(x)
        self.y = np.array(y)
        self.z = np.array(z)

        self.n = self.x.shape[0]

        self.xi = np.zeros(self.n)
        self.yi = np.zeros(self.n)
        self.zi = np.zeros(self.n)

        self.x0 = None
        self.y0 = None
        self.z0 = None
        self.x0i = None
        self.y0i = None
        self.z0i = None



        # set the serach elipse and variogram rotations-ranges as an array
        if a is None:
            # only use one rotation, that one of the search ellipses
            self.a = np.ones([1,3]) # 1 unit search ellipse
        else:
            assert np.array(a).shape[1] == 3
            self.a = np.array(a)

        # set the pibot point
        if pivot is None:
            self.pivot = np.zeros(3)
        else:
            assert pivot.shape[0] == (3,)
            self.pivot = pivot

        # set the pibot point
        if rot is None:
            self.rot = np.zeros(self.a.shape)
        else:
            assert np.array(rot).shape == self.a.shape
            self.rot = np.array(rot)


        self.sina = np.sin(self.rot[0,0]*DEG2RAD)
        self.sinb = np.sin(self.rot[0,1]*DEG2RAD)
        self.sint = np.sin(self.rot[0,2]*DEG2RAD)
        self.cosa = np.cos(self.rot[0,0]*DEG2RAD)
        self.cosb = np.cos(self.rot[0,1]*DEG2RAD)
        self.cost = np.cos(self.rot[0,2]*DEG2RAD)

        if test is None:
             self.test = np.zeros(self.n, dtype = bool)
        else:
             assert self.n.shape == (test.shape[0],)
             self.test = test

        # isotropic distance
        self.sqdisti = np.zeros(self.n)

        # row ID, will no chanege
        self.row_id = np.arange(self.n)

        # translate
        X1 = self.x - self.pivot[0]
        Y1 = self.y - self.pivot[1]
        Z1 = self.z - self.pivot[2]

        # rotate see http://www.ccgalberta.com/ccgresources/report06/2004-403-angle_rotations.pdf
        X2 = (self.cosa*self.cost+self.sina*self.sinb*self.sint)*X1 + \
             (-self.sina*self.cost+self.cosa*self.sinb*self.sint)*Y1 + \
             (-self.cosb*self.sint)*Z1
        Y2 = (self.sina*self.cosb)*X1 + \
             (self.cosa*self.cosb)*Y1 + \
             (self.sinb)*Z1
        Z2 = (self.cosa*self.sint-self.sina*self.sinb*self.cost)*X1 + \
             (-self.sina*self.sint-self.cosa*self.sinb*self.cost)*Y1 + \
             (self.cosb*self.cost)*Z1

        # rescale
        self.xi= X2
        self.yi= Y2/(self.a[0,1]/self.a[0,0]) # first row is for search ellipse
        self.zi= Z2/(self.a[0,2]/self.a[0,0])


    def update(self,t0):
        self.x0 = t0[0]
        self.y0 = t0[1]
        self.z0 = t0[2]

        # translate target point
        X1 = self.x0 - self.pivot[0]
        Y1 = self.y0 - self.pivot[1]
        Z1 = self.z0 - self.pivot[2]

        # rotate see http://www.ccgalberta.com/ccgresources/report06/2004-403-angle_rotations.pdf
        X2 = (self.cosa*self.cost+self.sina*self.sinb*self.sint)*X1 + \
             (-self.sina*self.cost+self.cosa*self.sinb*self.sint)*Y1 + \
             (-self.cosb*self.sint)*Z1
        Y2 = (self.sina*self.cosb)*X1 + \
             (self.cosa*self.cosb)*Y1 + \
             (self.sinb)*Z1
        Z2 = (self.cosa*self.sint-self.sina*self.sinb*self.cost)*X1 + \
             (-self.sina*self.sint-self.cosa*self.sinb*self.cost)*Y1 + \
             (self.cosb*self.cost)*Z1

        # rescale
        self.x0i= X2
        self.y0i= Y2/(self.a[0,1]/self.a[0,0])
        self.z0i= Z2/(self.a[0,2]/self.a[0,0])

        # test
        self.sqdisti = (self.xi - self.x0i)**2 + (self.yi - self.y0i)**2 + (self.zi - self.z0i)**2
        # self.sqdist =  (self.x - self.x0)**2 + (self.y - self.y0)**2 + (self.z - self.z0)**2
        self.test = self.sqdisti<=self.a[0,0]**2

        # return self.row_id[self.test]


    def get_distances(self, rotID = None):

        # rotID == None, then return raw distance
        if rotID is None:
            x = self.x[self.test]
            y = self.y[self.test]
            z = self.z[self.test]

            return np.sqrt((x - self.x0)**2 + (y - self.y0)**2 + (z - self.z0)**2)

        # return isotropic distance relative to dearch
        if rotID==0:
            return np.sqrt(self.sqdisti[self.test])

        # return isotropic distance i
        if rotID>0:

            sina = np.sin(self.rot[rotID,0]*DEG2RAD)
            sinb = np.sin(self.rot[rotID,1]*DEG2RAD)
            sint = np.sin(self.rot[rotID,2]*DEG2RAD)
            cosa = np.cos(self.rot[rotID,0]*DEG2RAD)
            cosb = np.cos(self.rot[rotID,1]*DEG2RAD)
            cost = np.cos(self.rot[rotID,2]*DEG2RAD)

            # get points selected
            x = self.x[self.test]
            y = self.y[self.test]
            z = self.z[self.test]

            # translate data
            X1 = x - self.pivot[0]
            Y1 = y - self.pivot[1]
            Z1 = z - self.pivot[2]

            # rotate see http://www.ccgalberta.com/ccgresources/report06/2004-403-angle_rotations.pdf
            X2 = (cosa*cost+sina*sinb*sint)*X1 + \
                 (-sina*cost+cosa*sinb*sint)*Y1 + \
                 (-cosb*sint)*Z1
            Y2 = (sina*cosb)*X1 + \
                 (cosa*cosb)*Y1 + \
                 (sinb)*Z1
            Z2 = (cosa*sint-sina*sinb*cost)*X1 + \
                 (-sina*sint-cosa*sinb*cost)*Y1 + \
                 (cosb*cost)*Z1

            # rescale
            xi= X2
            yi= Y2/(self.a[rotID,1]/self.a[rotID,0])
            zi= Z2/(self.a[rotID,2]/self.a[rotID,0])

            # translate pibot points
            X1 = self.x0 - self.pivot[0]
            Y1 = self.y0 - self.pivot[1]
            Z1 = self.z0 - self.pivot[2]

            # rotate see http://www.ccgalberta.com/ccgresources/report06/2004-403-angle_rotations.pdf
            X2 = (cosa*cost+sina*sinb*sint)*X1 + \
                 (-sina*cost+cosa*sinb*sint)*Y1 + \
                 (-cosb*sint)*Z1
            Y2 = (sina*cosb)*X1 + \
                 (cosa*cosb)*Y1 + \
                 (sinb)*Z1
            Z2 = (cosa*sint-sina*sinb*cost)*X1 + \
                 (-sina*sint-cosa*sinb*cost)*Y1 + \
                 (cosb*cost)*Z1

            # rescale
            x0i= X2
            y0i= Y2/(self.a[rotID,1]/self.a[rotID,0])
            z0i= Z2/(self.a[rotID,2]/self.a[rotID,0])

            return np.sqrt((xi - x0i)**2 + (yi - y0i)**2 + (zi - z0i)**2)
