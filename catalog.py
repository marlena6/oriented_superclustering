from headers import *

##################################################################################
##################################################################################

class Catalog(object):

    def __init__(self, name="test", nameLong=None, pathInCatalog="",  nObj=None, fig_dir='/pscratch/sd/b/boryanah/ACTxDESI/figs/', cat_fn="/catalog.txt"):
        '''nObj: used to keep the first nObj objects of the catalog, useful for quick debugging
        '''

        self.name = name
        if nameLong is None:
            self.nameLong = self.name
        else:
            self.nameLong = nameLong
        self.pathInCatalog = pathInCatalog

        # Figures path
        self.pathFig = fig_dir+self.name
        if not os.path.exists(self.pathFig):
            os.makedirs(self.pathFig)
        
        
        self.loadCatalog(nObj=nObj)
   

   ##################################################################################
   ##################################################################################

    def readInputCatalog(self):
        print("- read input catalog from "+self.pathInCatalog)
        if ".csv" in self.pathInCatalog:
            header, data = read_csv_with_header(self.pathInCatalog)
            if self.nObj is None:
                self.nObj = len(data['RA'])
            # sky coordinates and redshift
            self.RA = data['RA'][:self.nObj] # [deg]
            self.DEC = data['DEC'][:self.nObj]  # [deg]
            self.Z = data['Z'][:self.nObj]
            self.cosa = data['ca'][:self.nObj] # cos(alpha)
            self.sina = data['sa'][:self.nObj] # sin(alpha)
            self.x_asym = data['x_asym'][:self.nObj] # 1 or -1
            self.y_asym = data['y_asym'][:self.nObj] # 1 or -1
            self.constraints = data['constraints'][:self.nObj] # 1 or -1
            self.hdr = header