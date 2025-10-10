import os
import utils
import numpy as np
##################################################################################
##################################################################################

class Catalog(object):

    def __init__(self, name="test", nameLong=None, pathInCatalog="",  nObj=None, cat_fn="/catalog.txt"):
        '''nObj: used to keep the first nObj objects of the catalog, useful for quick debugging
        '''

        self.name = name
        if nameLong is None:
            self.nameLong = self.name
        else:
            self.nameLong = nameLong
        self.pathInCatalog = pathInCatalog
        self.readInputCatalog(nObj=nObj)
   

   ##################################################################################
   ##################################################################################


    def readInputCatalog(self, nObj=None):
        print("- read input catalog from "+self.pathInCatalog)
        self.nObj = nObj
        if ".csv" in self.pathInCatalog:
            
            header, data = utils.read_csv_with_header(self.pathInCatalog)
            if self.nObj is None:
                self.nObj = len(data['RA'])
            # sky coordinates and redshift
            # make random selection of nObj
            rand_sel = np.random.choice(len(data['RA']), self.nObj, replace=False)
            self.RA = data['RA'][rand_sel].to_numpy() # [deg]
            self.DEC = data['DEC'][rand_sel].to_numpy()  # [deg]
            self.Z = data['Z'][rand_sel].to_numpy()
            self.alpha = data['alpha'][rand_sel].to_numpy() # cos(alpha)
            self.x_asym = data['x_asym'][rand_sel].to_numpy() # 1 or -1
            self.y_asym = data['y_asym'][rand_sel].to_numpy() # 1 or -1
            self.constraints = data['constraints'][rand_sel].to_numpy()
            self.hdr = header