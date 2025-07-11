import numpy as np
from mendeleev.fetch import fetch_table

from utils.utils import Z_Score

props_name = ['atomic_weight', 'atomic_radius', 'atomic_volume', 'dipole_polarizability',
              'fusion_heat', 'thermal_conductivity', 'vdw_radius', 'en_pauling']
dim_atomic_feat = len(props_name)

def load_atomic_props():
    tb = fetch_table('elements')

    # 둘 다 가능한게 밑에께 더 빠를듯 성능 차이는 없음
    # nums = tb['atomic_number'].values.astype(int)
    nums = np.array(tb['atomic_number'], dtype = int)

    # 둘 다 가능한게 밑에께 더 빠를듯 성능 차이는 없음
    # props = np.nan_to_num(tb[props_name].values.astype(float))
    props = np.nan_to_num(np.array(tb[props_name], dtype = float))

    props_zscore = Z_Score(props)

    # 둘 다 가능한게 성능 차이는 없음
    # props_dict = {num: props_zscore[i, :] for i, num in enumerate(nums)}
    props_dict = {nums[i]: props_zscore[i, :] for i in range(0, nums.shape[0])}

    return props_dict

props = load_atomic_props()

# 여긴 동일함
# 코드도 거의 동일, 성능도 동일