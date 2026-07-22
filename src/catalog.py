import utils
import numpy as np
from astropy.table import Table
import fitsio
##################################################################################
##################################################################################

class Catalog(object):

    def __init__(self, name="test", nameLong=None, pathInCatalog="",  nObj=None, config={}):
        '''nObj: used to keep the first nObj objects of the catalog, useful for quick debugging
        '''

        self.name = name
        if nameLong is None:
            self.nameLong = self.name
        else:
            self.nameLong = nameLong
        self.pathInCatalog = pathInCatalog
        self.readInputCatalog(nObj=nObj, config=config)
   

   ##################################################################################
   ##################################################################################


    def readInputCatalog(self, nObj=None, config={}):
        print("- read input catalog from "+self.pathInCatalog)
        self.nObj = nObj
        print(self.nObj)
        if ".csv" in self.pathInCatalog:
            header, data = utils.read_csv_with_header(self.pathInCatalog)
            colnames = data.columns
        elif ".fits" in self.pathInCatalog:
            data = Table(fitsio.read(self.pathInCatalog))
            colnames = data.colnames
            header = None
        else:
            raise ValueError("File type must be csv or fits, no other types yet implemented")
        if self.nObj is None:
            self.nObj = len(data['RA'])
        # sky coordinates and redshift
        sel = np.ones(len(data['RA'])).astype(bool)
        if self.nObj < len(data['RA']):
            # make random selection of nObj by setting remainder to false
            n_remove = len(data['RA']) - self.nObj
            remove_idx = np.random.choice(len(data['RA']),
                        size=n_remove,
                        replace=False)
    
            sel[remove_idx] = False
            print("Randomly selected "+str(self.nObj)+" objects from the catalog")
            print("Old catalog size: "+str(len(data['RA'])))
            print("New catalog size: "+str(np.sum(sel)))
        # check for any constraints
        nu_min = config.get('nu_min', None)
        nu_max = config.get('nu_max', None)
        e_min  = config.get('e_min', None)
        e_max  = config.get('e_max', None)

        if nu_min is not None or nu_max is not None:
            assert('nu' in colnames), "Data must have nu values to put a threshold."
        if e_min is not None or e_max is not None:
            assert('e' in colnames), "Data must have e values to put a threshold."    
        # load the nu and e data
        if 'nu' in colnames:
            self.nu = data['nu']
        if 'e' in colnames:
            self.e  = data['e']

        selections = []
        if nu_min is not None:
            selnumin = self.nu > nu_min
            selections.append(selnumin)
        if nu_max is not None:
            selnumax = self.nu < nu_max
            selections.append(selnumax)
        if e_min is not None:
            selemin = self.e > e_min
            selections.append(selemin)
        if e_max is not None:
            selemax = self.e < e_max
            selections.append(selemax)
        if len(selections)>0:
            for sel_i in selections:
                sel = sel & sel_i # check that this works

        self.RA = data['RA'][sel] # [deg]
        self.DEC = data['DEC'][sel]  # [deg]
        self.Z = data['Z'][sel]
        self.alpha = None
        self.vR = None
        self.x_asym = None
        self.y_asym = None
        self.e = None
        self.nu = None
        self.constraints = None
        
        # the following parameters are optional; only read if they exist in the catalog
        if 'alpha' in colnames:
            self.alpha = data['alpha'][sel] # cos(alpha)
        if 'x_asym' in colnames:
            self.x_asym = data['x_asym'][sel] # 1 or -1
        if 'y_asym' in colnames:
            self.y_asym = data['y_asym'][sel] # 1 or -1
        if 'constraints' in colnames:
            self.constraints = data['constraints'][sel]
        if 'nu' in colnames:
            self.nu = data['nu'][sel]
        if 'e' in colnames:
            self.e  = data['e'][sel]
        if 'vR' in colnames:
            self.vR = data['vR'][sel] # 1 or -1
        self.hdr = header
        # make sure everything is a simple numpy array
        for attr in ['RA', 'DEC', 'Z', 'alpha', 'vR', 'x_asym', 'y_asym', 'e', 'nu']:
            if getattr(self, attr) is not None:
                setattr(self, attr, np.array(getattr(self, attr)))
                

        
    


