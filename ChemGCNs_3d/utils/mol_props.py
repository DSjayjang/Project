import numpy as np
from mendeleev.fetch import fetch_table

from utils.utils import Z_Score

props_name = ['atomic_weight', 'atomic_radius', 'atomic_volume', 'dipole_polarizability',
              'fusion_heat', 'thermal_conductivity', 'vdw_radius', 'en_pauling']
dim_atomic_feat = len(props_name)

def load_atomic_props():
    tb = fetch_table('elements')
    nums = np.array(tb['atomic_number'], dtype = int)

    props = np.nan_to_num(np.array(tb[props_name], dtype = float))
    props_zscore = Z_Score(props)
    props_dict = {nums[i]: props_zscore[i, :] for i in range(0, nums.shape[0])}

    return props_dict

props = load_atomic_props()