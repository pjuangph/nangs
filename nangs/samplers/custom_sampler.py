from typing import Dict
from .base_sampler import BaseSampler
import torch
import numpy as np 

class Custom_Sampler(BaseSampler):
    def __init__(self, data:Dict[str,np.ndarray],n_samples:int=1, device="cpu"):
        """This is for when using custom points to define your variables. 
        
        For example: 
            If you wanted a 2D domain x=[0,1] y=[0,1] with dimensions nx and ny
            you would have x = 2D numpy array x.shape = (nx,ny) y.shape = (nx,ny)  
            this can be flattened so that x is (1,nx*ny) and passed into custom_sampler

        Args:
            data (Dict[str,np.ndarray]): [description]
            device (str, optional): [description]. Defaults to "cpu".
        """
        super().__init__(data, n_samples,device)
        for k,v in data.items():
            data[k] = torch.as_tensor(v,device=device,dtype=torch.float32)

    def sample(self, n_samples:int=None):
        """This returns the data as specified 

        Args:
            n_samples (int, optional): not used. Defaults to None.

        Returns:
            Dict[str, torch.Tensor]: dictionary containing original keys but values are converted to tensors
        """
        new_data = dict()
        for k,v in self.data.items():
            if len(v) == 0:
                new_data[k] = torch.ones((self.n_samples,1),device=self.device,dtype=torch.float32)*v
            else:
                new_data[k] =v 
        return new_data
