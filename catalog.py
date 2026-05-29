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
            if self.nObj < len(data['RA']):
                # make random selection of nObj
                sel = np.random.choice(len(data['RA']), self.nObj, replace=False)
            else:
                sel = np.arange(len(data['RA'])) # just index all objects, don't change ordering
            self.RA = data['RA'][sel].to_numpy() # [deg]
            self.DEC = data['DEC'][sel].to_numpy()  # [deg]
            self.Z = data['Z'][sel].to_numpy()
            self.alpha = None
            self.x_asym = None
            self.y_asym = None
            self.constraints = None
            # the following parameters are optional; only read if they exist in the catalog
            if 'alpha' in data:
                self.alpha = data['alpha'][sel].to_numpy() # cos(alpha)
            if 'x_asym' in data:
                self.x_asym = data['x_asym'][sel].to_numpy() # 1 or -1
            if 'y_asym' in data:
                self.y_asym = data['y_asym'][sel].to_numpy() # 1 or -1
            if 'constraints' in data:
                self.constraints = data['constraints'][sel].to_numpy()
            self.hdr = header
            

    