# %%
import matplotlib.pyplot as plt
import numpy as np
# %%
pos = 3
i = np.arange(512)[np.newaxis, :]
denominator = 1 / np.power(10000, (2 * i) / np.float32(512))
numerator = np.arange(pos)[:, np.newaxis]
numerator * denominator
# %%
# %%
