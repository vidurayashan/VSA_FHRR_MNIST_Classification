#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import math


# In[14]:


import lib.python.lib_FHRR_Loihi as lib
import lib.python.utility as util


# In[15]:


from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.run_configs import Loihi2HwCfg
from lava.magma.core.run_conditions import RunSteps

class FHRR_SNN_Model():
    
    def __init__(self, dimension, run_loihi=False):
        
        # if (run_loihi):
        #     import lib_FHRR_Loihi as lib
        #     print("Loihi Library Loaded!")
        # else:
        #     import lib_FHRR as lib
        
        self.dimension = dimension
        self.hd1 = np.zeros(dimension)
        self.hd2 = np.zeros(dimension)
        self.n1  = lib.FHRR_Encoder(vec = self.hd1)
        self.n2  = lib.FHRR_Encoder(vec = self.hd2)
        self.add = lib.FHRR_Sum(dimension=dimension)
        self.mul = lib.FHRR_Multiply(dimension=dimension)
        self.decode_mul   = lib.FHRR_Decoder(dimension=dimension)
        self.decode_add   = lib.FHRR_Decoder(dimension=dimension)
        
        self.n1.spikes_out.connect(self.mul.spikes_a_in)
        self.n2.spikes_out.connect(self.mul.spikes_b_in)
        self.mul.spikes_out.connect(self.decode_mul.spikes_a_in)
        
        self.n1.spikes_out.connect(self.add.spikes_a_in)
        self.n2.spikes_out.connect(self.add.spikes_b_in)
        self.add.spikes_out.connect(self.decode_add.spikes_a_in)
        
        self.run_loihi = run_loihi
        self.run_cfg = Loihi2HwCfg()
        
        # if (self.run_loihi):
        #     self.run_cfg = Loihi2HwCfg()
        
        # Just init the runtime 
        self.decode_mul.run(condition=
                            RunSteps(
                                num_steps=1), 
                            run_cfg=self.run_cfg)
        
        self.voltage_n1 = []
        self.voltage_n2 = []
        self.voltage_mult = []
        self.voltage_add  = []
        self.voltage_deco_mult = []
        self.voltage_deco_add = []
    
    '''
    Cosine similarity of [v2] relative to [v1]
    '''
    def similarity(self, v1, v2):
        if (len(v1) != len(v2)):
            assert("Dimensions do not match!")
        v1_phase = util.complexarr2phase(v1)
        v2_phase = util.complexarr2phase(v2)
        error_v = np.array([ math.cos(a-b) 
                            for a, b in zip(v1_phase, v2_phase) ])
        return np.sum(error_v) / len(v1)
    
    '''
    Binding using normal binding logic
    '''
    def bind_py(self, v1, v2):
        rslt = [ (a * b) for a, b in zip(v1, v2) ]
        return rslt
    
    def bundle_py(self, v1, v2):
        v1_norm  = util.normalize_cmplx(v1)
        v2_norm  = util.normalize_cmplx(v2)
        
        rslt = [ (a + b) for a, b in zip(v1_norm, v2_norm) ]
        return rslt
    
    def run(self, condition, run_cfg, debug=False):

        if (debug == False):
            self.decode_mul.run(condition=condition,run_cfg=run_cfg)
        else:
            for i in range(lib.GlobalVars.time_steps):
                self.voltage_n1.append(self.n1.get_v())
                self.voltage_n2.append(self.n2.get_v())
                self.voltage_mult.append(self.mul.get_v())
                self.voltage_add.append(self.add.get_v())
                self.voltage_deco_mult.append(self.decode_mul.get_v())
                self.voltage_deco_add.append(self.decode_add.get_v())
                self.decode_mul.run(condition=
                            RunSteps(num_steps=1), 
                            run_cfg=self.run_cfg)
                
        return self.voltage_n1, self.voltage_n2, self.voltage_mult, self.voltage_add, self.voltage_deco_mult, self.voltage_deco_add
        
    '''
    Binding using Lava SNN
    '''
    def bind(self, v1, v2, debug = False):
        self.n1.input_vec.set(np.array(util.complexarr2phase(v1)))
        self.n2.input_vec.set(np.array(util.complexarr2phase(v2)))
        
        self.n1.v.set(np.zeros(self.dimension))
        self.n2.v.set(np.zeros(self.dimension))
        self.mul.a_last_spk.set(np.zeros(self.dimension))
        self.mul.b_last_spk.set(np.zeros(self.dimension))
        self.mul.a_period.set(np.full((self.mul.a_period.shape), lib.GlobalVars.global_max))
        self.mul.b_period.set(np.full((self.mul.b_period.shape), lib.GlobalVars.global_max))
        
        self.decode_mul.v.set(np.zeros(self.dimension))
        
        voltage_n1, voltage_n2, voltage_mult, voltage_add, voltage_deco_mult, voltage_deco_add = self.run(condition=
                             RunSteps(num_steps=lib.GlobalVars.time_steps),
                             run_cfg=self.run_cfg, debug=debug)
                 
                 
        raw_rslt  = self.decode_mul.get_decoded_value()
        rect_rslt = util.phase2complex_array(raw_rslt)
        
        return rect_rslt, [voltage_n1, voltage_n2, voltage_mult, voltage_add, voltage_deco_mult, voltage_deco_add]
    
    '''
    Bundle using Lava SNN
    '''
    def bundle(self, v1, v2, debug = False):
        self.n1.input_vec.set(np.array(util.complexarr2phase(v1)))
        self.n2.input_vec.set(np.array(util.complexarr2phase(v2)))
        
        self.n1.v.set(np.zeros(self.dimension))
        self.n2.v.set(np.zeros(self.dimension))
        self.add.a_last_spk.set(np.zeros(self.dimension))
        self.add.b_last_spk.set(np.zeros(self.dimension))
        self.add.a_period.set(np.full((self.add.a_period.shape), lib.GlobalVars.global_max))
        self.add.b_period.set(np.full((self.add.b_period.shape), lib.GlobalVars.global_max))
        
        self.decode_add.v.set(np.zeros(self.dimension))
        
        voltage_n1, voltage_n2, voltage_mult, voltage_add, voltage_deco_mult, voltage_deco_add = self.run(condition=
                             RunSteps(num_steps=lib.GlobalVars.time_steps),
                             run_cfg=self.run_cfg, debug=debug)
                 
                 
        raw_rslt  = self.decode_add.get_decoded_value()
        rect_rslt = util.phase2complex_array(raw_rslt)
        
        return rect_rslt, [voltage_n1, voltage_n2, voltage_mult, voltage_add, voltage_deco_mult, voltage_deco_add]


# In[ ]:




