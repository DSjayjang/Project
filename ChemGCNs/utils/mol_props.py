import numpy as np
from mendeleev.fetch import fetch_table

from utils.utils import Z_Score

props_name = ['atomic_weight', 'atomic_radius', 'atomic_volume', 'dipole_polarizability',
              'fusion_heat', 'thermal_conductivity', 'vdw_radius', 'en_pauling']

def load_atomic_props():
    tb = fetch_table('elements')
    nums = tb['atomic_number'].values.astype(int)
    props = np.nan_to_num(tb[props_name].values.astype(float))
    props_zscore = Z_Score(props)
    props_dict = {num: props_zscore[i, :] for i, num in enumerate(nums)}

    return props_dict

props = load_atomic_props()