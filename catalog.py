import utils
import numpy as np
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
        if ".csv" in self.pathInCatalog:
            header, data = utils.read_csv_with_header(self.pathInCatalog)
            if self.nObj is None:
                self.nObj = len(data['RA'])
            # sky coordinates and redshift
            sel = np.ones(len(data['RA'])).astype(bool)
            if self.nObj < len(data['RA']):
                # make random selection of nObj by setting remainder to false
                sel[np.random.choice(len(data['RA'])-self.nObj, replace=False)] = False

            # check for any constraints
            nu_min = config.get('nu_min', None)
            nu_max = config.get('nu_max', None)
            e_min  = config.get('e_min', None)
            e_max  = config.get('e_max', None)

            if nu_min is not None or nu_max is not None:
                assert('nu' in data), "Data must have nu values to put a threshold."
            if e_min is not None or e_max is not None:
                assert('e' in data), "Data must have e values to put a threshold."    
            # load the nu and e data
            if 'nu' in data:
                self.nu = data['nu'].to_numpy()
            if 'e' in data:
                self.e  = data['e'].to_numpy()

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

            self.RA = data['RA'][sel].to_numpy() # [deg]
            self.DEC = data['DEC'][sel].to_numpy()  # [deg]
            self.Z = data['Z'][sel].to_numpy()
            self.alpha = None
            self.vR = None
            self.x_asym = None
            self.y_asym = None
            self.e = None
            self.nu = None
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
            if 'nu' in data:
                self.nu = data['nu'][sel].to_numpy()
            if 'e' in data:
                self.e  = data['e'][sel].to_numpy()
            if 'vR' in data:
                self.vR = data['vR'][sel].to_numpy() # 1 or -1
            self.hdr = header

            
        else:
            raise ValueError("File type must be csv, no other types yet implemented")

    
