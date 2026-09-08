import utils
import numpy as np
from astropy.table import Table, column
import fitsio
import copy
from pixell import enmap
##################################################################################
##################################################################################

class Catalog(object):

    def __init__(self, pathInCatalog, name="", nObj=None, config={}):
        '''nObj: used to keep the first nObj objects of the catalog, useful for quick debugging
        '''

        self.name = name
        self.pathInCatalog = pathInCatalog
        self.readInputCatalog(nObj=nObj, config=config)
   

   ##################################################################################
   ##################################################################################
    

    def readInputCatalog(self, nObj=None, config={}, remove_nans=True):
        print("- read input catalog from "+self.pathInCatalog)
        self.nObj = nObj
        
        possible_colnames = ['ra','dec','w','wgt','weight','z','alpha','x_pol','y_pol','nu','e','vr', 'id','targetid']
        if ".csv" in self.pathInCatalog:
            header, data = utils.read_csv_with_header(self.pathInCatalog)
            colnames = data.columns
            # convert dataframe to dict of numpy
            data = {col: data[col].to_numpy() for col in colnames}
        elif ".fits" in self.pathInCatalog:
            data = Table(fitsio.read(self.pathInCatalog))
            colnames = data.colnames
            header = None
            # conver table to dict of numpy
            data = {col: data[col].data for col in colnames}
        else:
            raise ValueError("File type must be csv or fits, no other types yet implemented")
        
        col_lookup = {key.lower(): key for key in colnames}
        ra_key = col_lookup['ra']
        if self.nObj is None:
            self.nObj = len(data[ra_key])
        # sky coordinates and redshift
        sel = np.ones(len(data[ra_key])).astype(bool)
        if self.nObj < len(data[ra_key]):
            # make random selection of nObj by setting remainder to false
            n_remove = len(data[ra_key]) - self.nObj
            remove_idx = np.random.choice(len(data[ra_key]),
                        size=n_remove,
                        replace=False)
    
            sel[remove_idx] = False
            print("Randomly selected "+str(self.nObj)+" objects from the catalog")
            print("Old catalog size: "+str(len(data[ra_key])))
            print("New catalog size: "+str(np.sum(sel)))
        # check for any constraints
        nu_min = config.get('nu_min', None)
        nu_max = config.get('nu_max', None)
        e_min  = config.get('e_min', None)
        e_max  = config.get('e_max', None)

        if nu_min is not None or nu_max is not None:
            assert 'nu' in colnames, "Data must have nu values to put a threshold."
        if e_min is not None or e_max is not None:
            assert 'e' in colnames, "Data must have e values to put a threshold."    
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
        for key in colnames:
            if key.lower() in possible_colnames:
                if key.lower() in ['id', 'targetid']:
                    setattr(self, 'id', data[key][sel].astype(np.int64)) # this is often a long integer
                elif key.lower() in ['wgt','w','weight']:
                    setattr(self, 'w', data[key][sel])
                else:
                    setattr(self, key.lower(), data[key][sel])

        # special treatment for w
        if not hasattr(self, 'w') or self.w is None:
            print("No weights found in catalog. Setting all weights to 1.")
            self.w = np.ones(len(self.ra)) # set weights to 1
        
        self.hdr = header # this can contain constraints applied, other metadata, etc.

        # make sure every list is a numpy array
        for attr, value in self.__dict__.items():
            if isinstance(value, (list, column.Column)):
                setattr(self, attr, np.asarray(value))
            
        if remove_nans:
            self.remove_nans()
            

    def remove_nans(self):
        good = np.ones(len(self.ra)).astype(bool)
        for attr in ['ra', 'dec', 'z', 'alpha', 'vr', 'x_pol', 'y_pol', 'e', 'nu', 'w']:
            if attr in self.__dict__.keys():
                idx_good = ~np.isnan(getattr(self, attr))
                good &= idx_good

        if sum(~good) > 0:
            for attr in ['id', 'ra', 'dec', 'z', 'alpha', 'vr', 'x_pol', 'y_pol', 'e', 'nu', 'w']:
                if attr in self.__dict__.keys():
                    setattr(self, attr, getattr(self, attr)[good])
            self.nObj = len(self.ra) # reset nObj
            print("Removed "+str(sum(~good))+" objects with NaN values from the catalog.")
            print("New catalog size: "+str(self.nObj))
    
    def prune_catalog(self, index=None, condition=None, inplace=False):
        """Reduce the catalog to a set of indices or to satisfy particular conditions.

        Args:
            index (array-like): Indices of the catalog to keep. If provided, condition
                is ignored.

            condition (dict): Dictionary of conditions to apply to the catalog.
                Keys are column names (case-insensitive) and values are tuples of
                (min, max) values. If min or max is None, that bound is ignored.

                Example:
                    {'Z': (0.1, 0.5), 'NU': (None, 10)}

                keeps objects with 0.1 < z < 0.5 and nu < 10.

            return_new_catalog (bool): If True, return a new catalog without
                modifying the original. Only the catalog arrays are copied.

        Returns:
            If return_new_catalog is True, the pruned catalog.
            Otherwise, None.
        """
        if condition is None:
            condition = {}

        attrs = [
            'ra', 'dec', 'z', 'alpha', 'vr',
            'x_pol', 'y_pol', 'e', 'nu', 'w', 'id'
        ]

        # Case-insensitive mapping from attribute name -> actual attribute name.
        attr_lookup = {attr.lower(): attr for attr in attrs}

        # Work on a new object or on self.
        if not inplace:
            catalog = copy.copy(self)
        else:
            catalog = self

        # ---------------------------------------------------------
        # Determine selection
        # ---------------------------------------------------------
        if index is not None:
            # Explicit indices
            selection = index

        else:
            # Conditions
            selection = np.ones(len(self.ra), dtype=bool)

            for key, (min_val, max_val) in condition.items():
                key_lower = key.lower()

                if key_lower not in attr_lookup:
                    raise ValueError(
                        f"Catalog does not have attribute '{key}'"
                    )

                attr = attr_lookup[key_lower]
                attr_values = getattr(self, attr)

                if attr_values is None:
                    raise ValueError(
                        f"Catalog attribute '{attr}' is None"
                    )

                if min_val is not None:
                    selection &= attr_values > min_val

                if max_val is not None:
                    selection &= attr_values < max_val

        # ---------------------------------------------------------
        # Apply selection
        # ---------------------------------------------------------
        for attr in attrs:
            value = getattr(self, attr, None)

            if value is not None:
                if not inplace:
                    # Copy only the selected portion of each array.
                    setattr(catalog, attr, value[selection].copy())
                else:
                    # Modify the existing catalog in-place.
                    setattr(catalog, attr, value[selection])

        if inplace:
            self.nObj = len(catalog.ra)  # Update nObj for the catalog
        if not inplace:
            return catalog  

    def sort_catalog(self, sort_by='z', ascending=True):
        """Sort the catalog by a specified attribute.

        Args:
            sort_by (str): Attribute name to sort by (case-insensitive).
            ascending (bool): If True, sort in ascending order; otherwise, descending.
        """
        attrs = [
            'ra', 'dec', 'z', 'alpha', 'vr',
            'x_pol', 'y_pol', 'e', 'nu', 'w', 'id'
        ]

        # Case-insensitive mapping from attribute name -> actual attribute name.
        attr_lookup = {attr.lower(): attr for attr in attrs}

        sort_by_lower = sort_by.lower()

        if sort_by_lower not in attr_lookup:
            raise ValueError(
                f"Catalog does not have attribute '{sort_by}'"
            )

        attr = attr_lookup[sort_by_lower]
        values = getattr(self, attr)

        if values is None:
            raise ValueError(
                f"Catalog attribute '{attr}' is None"
            )

        # Get the sorted indices
        sorted_indices = np.argsort(values)

        if not ascending:
            sorted_indices = sorted_indices[::-1]

        # Apply sorting to all attributes
        for attr in attrs:
            value = getattr(self, attr, None)

            if value is not None:
                setattr(self, attr, value[sorted_indices])

    def mask_catalog(self, imask, threshold=0.9):
        """ Use an enmap to reduce the catalog
        to only sources which overlap with where the mask > threshold."""
        
        # extract map values at ra,dec
        val = enmap.at(imask, np.deg2rad([self.dec, self.ra]), mode="nn")
        full_sample = val > threshold
        for attr in ['ra', 'dec', 'z', 'alpha', 'vr', 'x_pol', 'y_pol', 'e', 'nu', 'w', 'id']:
            if attr in self.__dict__.keys():
                setattr(self, attr, getattr(self, attr)[full_sample])