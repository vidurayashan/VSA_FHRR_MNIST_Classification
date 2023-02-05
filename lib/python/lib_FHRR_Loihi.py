#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort


# In[2]:


# import nbimporter
import lib.python.utility as util


# In[3]:


class GlobalVars():
    global_threshold = 10
    global_max = 1000000
    time_steps = global_threshold * 2 + 1


# In[4]:


class FHRR_Encoder(AbstractProcess):
    def __init__(self, vec : [complex]):
        super().__init__()
        hd_vec = util.complexarr2phase(vec)
        shape = (len(hd_vec),)
        self.spikes_out = OutPort(shape=shape)  # Input spikes to the classifier
        self.input_vec = Var(shape=shape, init=hd_vec)
        self.v = Var(shape=shape, init=0)
        self.vth = Var(shape=(1,), init=GlobalVars.global_threshold)
        
    def get_v(self):
        return self.v.get()

class FHRR_Decoder(AbstractProcess):

    def __init__(self, dimension : int):
        super().__init__()
        dim = dimension
        shape = (dim,)
        self.spikes_a_in = InPort(shape=shape)
        self.v = Var(shape=shape, init=0)
        self.decoded_a = Var(shape=shape, init=0)
        self.vth = Var(shape=(1,), init=GlobalVars.global_threshold)
        
        self.a_last_spk = Var(shape=shape, init=0)
        self.a_period = Var(shape=shape, init=GlobalVars.global_max)
        
    def get_v(self):
        return self.v.get()
    
    def get_decoded_value(self):
        return self.decoded_a.get()
    
class FHRR_Sum(AbstractProcess):

    def __init__(self, dimension : int):
        super().__init__()
        self.dim = Var(shape=(1,), init=dimension)
        shape = (dimension,)
        self.spikes_a_in = InPort(shape=shape)
        self.spikes_b_in = InPort(shape=shape)
        self.spikes_out = OutPort(shape=shape)
        self.v = Var(shape=shape, init=0)
        self.vth = Var(shape=(1,), init=GlobalVars.global_threshold)
        
        self.a_last_spk = Var(shape=shape, init=0)
        self.b_last_spk = Var(shape=shape, init=0)
        self.a_period = Var(shape=shape, init=GlobalVars.global_max)
        self.b_period = Var(shape=shape, init=GlobalVars.global_max)
        self.phase_arr_clean = Var(shape=shape, init=0)
        
    def get_v(self):
        return self.v.get()
    
    def get_phase_arr_clean(self):
        return self.phase_arr_clean.get()
    
class FHRR_Multiply(AbstractProcess):

    def __init__(self, dimension : int):
        super().__init__()
        self.dim = Var(shape=(1,), init=dimension)
        shape = (dimension,)
        self.spikes_a_in = InPort(shape=shape)
        self.spikes_b_in = InPort(shape=shape)
        self.spikes_out = OutPort(shape=shape)
        self.v = Var(shape=shape, init=0)
        self.vth = Var(shape=(1,), init=GlobalVars.global_threshold)
        
        self.a_last_spk = Var(shape=shape, init=0)
        self.b_last_spk = Var(shape=shape, init=0)
        self.a_period = Var(shape=shape, init=GlobalVars.global_max)
        self.b_period = Var(shape=shape, init=GlobalVars.global_max)
        
    def get_v(self):
        return self.v.get()


# In[5]:


import numpy as np
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU, LMT, OheoGulch, Loihi2NeuroCore
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel

@implements(proc=FHRR_Encoder, protocol=LoihiProtocol)
@requires(Loihi2NeuroCore)
class PySpikeInputModel(PyLoihiProcessModel):
    spikes_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
    input_vec: np.ndarray = LavaPyType(np.ndarray, float, precision=32)
    v: np.ndarray = LavaPyType(np.ndarray, int, precision=32)
    vth: int = LavaPyType(int, int, precision=32)
    
    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.time_step = 0

    def post_guard(self):
        """Guard function for PostManagement phase.
        """
        if self.time_step == 1:
            return True
        return False

    def run_post_mgmt(self):
        """Post-Management phase: executed only when guard function above 
        returns True.
        """
        self.v = np.zeros(self.v.shape)

    def run_spk(self):
        """Spiking phase: executed unconditionally at every time-step
        """
        self.v[:] = self.v + self.input_vec
        s_out = self.v > self.vth
        self.v[s_out] = 0  # reset voltage to 0 after a spike
        self.spikes_out.send(s_out)

@implements(proc=FHRR_Sum, protocol=LoihiProtocol)
@requires(Loihi2NeuroCore)
class PySpikeSumModel(PyLoihiProcessModel):   
    dim: int = LavaPyType(int, int, precision=32)
    spikes_a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    spikes_b_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)    
    spikes_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
    v: np.ndarray = LavaPyType(np.ndarray, float, precision=32)
    vth: int = LavaPyType(int, int, precision=32)
    phase_arr_clean: np.ndarray = LavaPyType(np.ndarray, float, precision=32)
    

    a_last_spk: np.ndarray = LavaPyType(np.ndarray, int, precision=32)
    b_last_spk: np.ndarray = LavaPyType(np.ndarray, int, precision=32)
    a_period: np.ndarray = LavaPyType(np.ndarray, float, precision=32)
    b_period: np.ndarray = LavaPyType(np.ndarray, float, precision=32)
    
    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.time_step = 0

    def post_guard(self):
        """Guard function for PostManagement phase.
        """
        if self.time_step == 1:
            return True
        return False

    def run_post_mgmt(self):
        """Post-Management phase: executed only when guard function above 
        returns True.
        """
#         print("Run Post Management")
        self.v          = np.zeros((self.v.shape))
        self.a_last_spk = np.zeros((self.v.shape))
        self.b_last_spk = np.zeros((self.v.shape))

    def run_spk(self):
        """Spiking phase: executed unconditionally at every time-step
        """
#         print(f"time : {self.time_step}")
        vec_time_step = np.full((self.v.shape), self.time_step)
        
        new_spike_times = self.spikes_a_in.peek() * vec_time_step 
        new_spikes      = (new_spike_times > 0) * 1
        new_spikes_inv  = 1 - (new_spikes > 0)
        
        masked_last_spike = self.a_last_spk * new_spikes
        self.a_period[:] = (new_spike_times - masked_last_spike) + new_spikes_inv * self.a_period
        
        masked_last_spike_inv = self.a_last_spk * new_spikes_inv
        self.a_last_spk = masked_last_spike_inv + new_spikes * vec_time_step
        
        decoded_a = GlobalVars.global_threshold / self.a_period
        complex_arr_a = util.phase2complex_array(decoded_a)
        
        ###############################################################################
        
        new_spike_times = self.spikes_b_in.peek() * vec_time_step 
        new_spikes      = (new_spike_times > 0) * 1
        new_spikes_inv  = 1 - (new_spikes > 0)
        
        masked_last_spike = self.b_last_spk * new_spikes
        self.b_period[:] = (new_spike_times - masked_last_spike) + new_spikes_inv * self.b_period
        
        masked_last_spike_inv = self.b_last_spk * new_spikes_inv
        self.b_last_spk = masked_last_spike_inv + new_spikes * vec_time_step
        
        decoded_b = GlobalVars.global_threshold / self.b_period
        complex_arr_b = util.phase2complex_array(decoded_b)
        
        ##############################################################################################

#         print(f"decoded_a         : {[item * 180 / cmath.pi for item in decoded_a]}")
#         print(f"decoded_b         : {[item * 180 / cmath.pi for item in decoded_b]}")
        
        sum_complex = [a + b for a,b in zip(complex_arr_a, complex_arr_b)]
        
        phase_arr = util.complexarr2phase(sum_complex)
        self.phase_arr_clean = util.cleanphase(phase_arr)
        
#         print(f"phase_arr_clean    : {[item * 180 / cmath.pi for item in self.phase_arr_clean]}")
        
        self.v[:] = self.v + self.phase_arr_clean
        s_out = self.v > self.vth
        self.v[s_out] = 0  # reset voltage to 0 after a spike
        self.spikes_out.send(s_out)
        

@implements(proc=FHRR_Multiply, protocol=LoihiProtocol)
@requires(Loihi2NeuroCore)
class PySpikeMultModel(PyLoihiProcessModel):   
    dim: int = LavaPyType(int, int, precision=32)
    spikes_a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    spikes_b_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)    
    spikes_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
    v: np.ndarray = LavaPyType(np.ndarray, float, precision=32)
    vth: int = LavaPyType(int, int, precision=32)
    

    a_last_spk: np.ndarray = LavaPyType(np.ndarray, int, precision=32)
    b_last_spk: np.ndarray = LavaPyType(np.ndarray, int, precision=32)
    a_period: np.ndarray = LavaPyType(np.ndarray, float, precision=32)
    b_period: np.ndarray = LavaPyType(np.ndarray, float, precision=32)
    
    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.time_step = 0

    def post_guard(self):
        """Guard function for PostManagement phase.
        """
        if self.time_step == 1:
            return True
        return False

    def run_post_mgmt(self):
        """Post-Management phase: executed only when guard function above 
        returns True.
        """
        self.v          = np.zeros((self.v.shape))
        self.a_last_spk = np.zeros((self.v.shape))
        self.b_last_spk = np.zeros((self.v.shape))


    def run_spk(self):
        """Spiking phase: executed unconditionally at every time-step
        """
#         print(f"time : {self.time_step}")
        vec_time_step = np.full((self.v.shape), self.time_step)
        
        new_spike_times = self.spikes_a_in.peek() * vec_time_step 
        new_spikes      = (new_spike_times > 0) * 1
        new_spikes_inv  = 1 - (new_spikes > 0)
        
        masked_last_spike = self.a_last_spk * new_spikes
        self.a_period[:] = (new_spike_times - masked_last_spike) + new_spikes_inv * self.a_period
        

        masked_last_spike_inv = self.a_last_spk * new_spikes_inv
        self.a_last_spk = masked_last_spike_inv + new_spikes * vec_time_step
        
        decoded_a = GlobalVars.global_threshold / self.a_period
        
        ###############################################################################
        
        new_spike_times = self.spikes_b_in.peek() * vec_time_step 
        new_spikes      = (new_spike_times > 0) * 1
        new_spikes_inv  = 1 - (new_spikes > 0)
        
        masked_last_spike = self.b_last_spk * new_spikes
        self.b_period[:] = (new_spike_times - masked_last_spike) + new_spikes_inv * self.b_period
        
        masked_last_spike_inv = self.b_last_spk * new_spikes_inv
        self.b_last_spk = masked_last_spike_inv + new_spikes * vec_time_step
        
        decoded_b = GlobalVars.global_threshold / self.b_period
        
        ##############################################################################################
        
#         print(f"decoded_a         : {[item * 180 / cmath.pi for item in decoded_a]}")
#         print(f"decoded_b         : {[item * 180 / cmath.pi for item in decoded_b]}")
        
        mult_phase = [a + b for a,b in zip(decoded_a, decoded_b)]
        
        mult_phase_cleaned = util.cleanphase(mult_phase)
        
#         print(f"mult_phase_cleaned         : {[item * 180 / cmath.pi for item in mult_phase_cleaned]}")
        
        self.v[:] = self.v + mult_phase_cleaned
        s_out = self.v > self.vth
        self.v[s_out] = 0
        self.spikes_out.send(s_out)
        


# In[6]:


@implements(proc=FHRR_Decoder, protocol=LoihiProtocol)
@requires(Loihi2NeuroCore)
class PySpikeDecoderModel(PyLoihiProcessModel):        
    spikes_a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, bool, precision=1)
    v: np.ndarray = LavaPyType(np.ndarray, float, precision=32)
    decoded_a: np.ndarray = LavaPyType(np.ndarray, float, precision=32)
    vth: int = LavaPyType(int, int, precision=32)

    a_last_spk: np.ndarray = LavaPyType(np.ndarray, int, precision=32)
    a_period: np.ndarray = LavaPyType(np.ndarray, float, precision=32)
    
    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.time_step = 0

    def post_guard(self):
        """Guard function for PostManagement phase.
        """
        if self.time_step == GlobalVars.time_steps - 1:
            return True
        return False

    def run_post_mgmt(self):
        """Post-Management phase: executed only when guard function above 
        returns True.
        """
#         print(f"decoded_val  : {[item * 180 / cmath.pi for item in self.decoded_a]}")

    def run_spk(self):
        """Spiking phase: executed unconditionally at every time-step
        """
        
        ###############################################################################
        
        vec_time_step = np.full((self.v.shape), self.time_step)
        
        new_spike_times = self.spikes_a_in.peek() * vec_time_step 
        new_spikes      = (new_spike_times > 0) * 1
        new_spikes_inv  = 1 - (new_spikes > 0)
        
        masked_last_spike = self.a_last_spk * new_spikes
        self.a_period[:] = (new_spike_times - masked_last_spike) + new_spikes_inv * self.a_period
        

        masked_last_spike_inv = self.a_last_spk * new_spikes_inv
        self.a_last_spk = masked_last_spike_inv + new_spikes * vec_time_step
        
        self.decoded_a = GlobalVars.global_threshold / self.a_period
        
#         print(f"decoded_val         : {[item * 180 / cmath.pi for item in self.decoded_a]}")
#         print(f'a_period            : {self.a_period}')
        
        self.v[:] = self.v + self.decoded_a
        s_out = self.v > self.vth
        self.v[s_out] = 0
        

