
# coding: utf-8

# In[1]:

import numpy as np
import sys
import time
sys.path.append('/Users/dangoldberg/Desktop/code/tlrl')
sys.path.append('/Users/dangoldberg/Desktop/code/tlrl/lib')

# In[12]:

from envs import *

threeD = ENVS_DICTIONARY['3DMountainCar'](trailer=True, show_velo=True)
twoD = ENVS_DICTIONARY['2DMountainCar'](trailer=True)


# In[13]:

source_trajectory = np.load('visualize_trajectories/2DMC-3DMC/source_trajectory.p')
mapped_trajectory = np.load('visualize_trajectories/2DMC-3DMC/bigbatch_trajectory.p')


# In[14]:

print('starting render')

for t in range(5000):
    
    twoD.state = source_trajectory[t]
    twoD.last_few_positions.append((source_trajectory[t][0]))
    if len(twoD.last_few_positions) == twoD.trail_num+1:
        del twoD.last_few_positions[0]
    twoD._render(action_vec=source_trajectory[t][4:])

    threeD.state = mapped_trajectory[t]
    threeD.last_few_positions.append((mapped_trajectory[t][0], mapped_trajectory[t][1]))
    if len(threeD.last_few_positions) == threeD.trail_num+1:
        del threeD.last_few_positions[0]
    threeD.render_orthographic(action_vec=mapped_trajectory[t][8:])
    
    time.sleep(0.0)
    
threeD.close()
twoD.close()


# In[ ]:


print('done')

# In[ ]:



