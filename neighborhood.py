import numpy as np
import concurrent.futures

DEG2RAD=3.141592654/180.0
EPSLON=1.e-20

def rotoscale(x,y,z,x0,y0,z0, ang1=0,ang2=0,ang3=0,anis1=1,anis2=1, inverse = False):
    """
    TODO:
    """
    sina = np.sin(ang1*DEG2RAD)
    sinb = np.sin(ang2*DEG2RAD)
    sint = np.sin(ang3*DEG2RAD)
    cosa = np.cos(ang1*DEG2RAD)
    cosb = np.cos(ang2*DEG2RAD)
    cost = np.cos(ang3*DEG2RAD)
    
    if not inverse:
        #translate
        x1 = x-x0
        y1 = y-y0
        z1 = z-z0
        # rotate using equation 4 on http://www.ccgalberta.com/ccgresources/report06/2004-403-angle_rotations.pdf
        x2= (cosa*cost+sina*sinb*sint)*x1 + (-sina*cost+cosa*sinb*sint)*y1 +(-cosb*sint)*z1
        y2=                (sina*cosb)*x1 +                 (cosa*cosb)*y1 +      (sinb)*z1
        z2= (cosa*sint-sina*sinb*cost)*x1 + (-sina*sint-cosa*sinb*cost)*y1 + (cosb*cost)*z1
        # rescale
        xr= x2
        yr= y2/anis1
        zr= z2/anis2
    else:  
        # rescale
        x1 = x
        y1 = y*anis1
        z1 = z*anis2
        # rotate using equation 5 on http://www.ccgalberta.com/ccgresources/report06/2004-403-angle_rotations.pdf
        x2= (cosa*cost+sina*sinb*sint)*x1 + (sina*cosb)*y1 + (cosa*sint-sina*sinb*cost)*z1
        y2=(-sina*cost+cosa*sinb*sint)*x1 + (cosa*cosb)*y1 +(-sina*sint-cosa*sinb*cost)*z1
        z2=               (-cosb*sint)*x1 +      (sinb)*y1 +                (cosb*cost)*z1
        # shift 
        xr = x2+x0
        yr = y2+y0
        zr = z2+z0
      
    return xr,yr,zr
        

class Neighborhood():
    """
    TODO:
    """

    def __init__(self,x,y,z,v = None, vname = None, vmeta = None, a= None, pivot = None, rot=None,test=None):

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

        # variables 
        self.v = {}
        self.vmeta = {}
        if v is not None:
            assert np.array(x).shape[0] == np.array(v).shape[0]
            for i in range(np.array(v).shape[1]):
                self.v[vname[i]] = v[:,i]
                if vmeta is not None:
                    self.vmeta[vname[i]] = vmeta[i]
                else:
                    self.vmeta[vname[i]] = None

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

        if test is None:
             self.test = np.zeros(self.n, dtype = bool)
        else:
             assert self.n.shape == (test.shape[0],)
             self.test = test

        # isotropic distance
        self.sqdisti = np.zeros(self.n)

        # row ID, will no chanege
        self.row_id = np.arange(self.n)
        
        self.xi, self.yi, self.zi = rotoscale(
                                        x = self.x,y = self.y,z = self.z,
                                        ang1  = self.rot[0,0], ang2 = self.rot[0,1],
                                        ang3  = self.rot[0,2],
                                        anis1 = self.a[0,1]/self.a[0,0],
                                        anis2 = self.a[0,2]/self.a[0,0],
                                        x0 = self.pivot[0], y0 = self.pivot[1],
                                        z0 = self.pivot[2], inverse = False)


    def update(self,t0):
        self.x0 = t0[0]
        self.y0 = t0[1]
        self.z0 = t0[2]

        self.x0i, self.y0i, self.z0i = rotoscale(
                                        x = self.x0,y = self.y0,z = self.z0,
                                        ang1  = self.rot[0,0], ang2 = self.rot[0,1],
                                        ang3  = self.rot[0,2],
                                        anis1 = self.a[0,1]/self.a[0,0],
                                        anis2 = self.a[0,2]/self.a[0,0],
                                        x0 = self.pivot[0], y0 = self.pivot[1],
                                        z0 = self.pivot[2], inverse = False)
        

        # test
        self.sqdisti = (self.xi - self.x0i)**2 + (self.yi - self.y0i)**2 + (self.zi - self.z0i)**2
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
            
            # get data points selected
            x = self.x[self.test]
            y = self.y[self.test]
            z = self.z[self.test]
            
            # rotate and rescale
            xi, yi, zi = rotoscale(
                            x = x,y = y,z = z,
                            ang1  = self.rot[rotID,0], ang2 = self.rot[rotID,1],
                            ang3  = self.rot[rotID,2],
                            anis1 = self.a[rotID,1]/self.a[rotID,0],
                            anis2 = self.a[rotID,2]/self.a[rotID,0],
                            x0 = self.pivot[0], y0 = self.pivot[1],
                            z0 = self.pivot[2], inverse = False)
            
  

            # rotate and rescale pibot points
            x0i, y0i, z0i = rotoscale(
                            x = self.x0,y = self.x0,z = self.x0,
                            ang1  = self.rot[rotID,0], ang2 = self.rot[rotID,1],
                            ang3  = self.rot[rotID,2],
                            anis1 = self.a[rotID,1]/self.a[rotID,0],
                            anis2 = self.a[rotID,2]/self.a[rotID,0],
                            x0 = self.pivot[0], y0 = self.pivot[1],
                            z0 = self.pivot[2], inverse = False)

            # return distance
            return np.sqrt((xi - x0i)**2 + (yi - y0i)**2 + (zi - z0i)**2)
