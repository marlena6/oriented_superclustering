# Pass in a path with many randoms file, make a single output randoms file including all
import sys
sys.path.insert(0,"/global/cfs/projectdirs/act/data/mlokken/oriented_stacks/oriented_superclustering/")
import glob
import os
import select_and_orient as sao
from astropy.table import Table

nrandoms = int(sys.argv[1]) # number of randoms per file, e.g. 10
outpath = sys.argv[2]
# randoms_path = "/mnt/raid-cita/mlokken/data/desi/randoms/LRG_*_clustering.ran.fits"
randoms_path = "/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/LRG_*_clustering.ran.fits"
randoms = glob.glob(randoms_path)

ra_rand = []
dec_rand = []
z_rand = []
w_rand = []
ids = []
N_randoms_files = 0
for ranfile in randoms:
    if N_randoms_files < nrandoms:
        if 'comb' not in ranfile and 'SGC' not in ranfile and 'NGC' not in ranfile:
            ra_rand_i, dec_rand_i, z_rand_i, id, w_rand_i = sao.get_radecz(ranfile, return_weight=True, return_id=True)
            ra_rand.extend(ra_rand_i)
            dec_rand.extend(dec_rand_i)
            z_rand.extend(z_rand_i)
            w_rand.extend(w_rand_i)
            ids.extend(id)
            N_randoms_files += 1

# newfile = f"/mnt/raid-cita/mlokken/data/desi/randoms/LRG_comb{N_randoms_files}_clustering.ran.fits"

new_table = Table([ra_rand, dec_rand, z_rand, ids, w_rand], names=['RA', 'DEC', 'Z', 'TARGETID', 'WEIGHT'])
new_table.write(os.path.join(outpath, f"LRG_comb{N_randoms_files}_clustering.ran.fits"), overwrite=True)