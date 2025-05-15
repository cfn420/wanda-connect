import torch
import torch.nn as nn
import torch.nn.functional as F

# Define WrappedGPT class
class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none", batch_size=1):
        self.layer = layer
        self.layer_name = layer_name
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]
        self.batch_size = batch_size

        self.scaler_row = torch.zeros((self.columns), device=self.dev)  # Tracks input statistics
        self.scaler_col = torch.zeros((self.rows), device=self.dev)     # Tracks output statistics

        if layer_name == 'mlp.gate_proj': 
            self.scaler_col_act = torch.zeros((self.rows), device=self.dev)
            self.scaler_col_latest = torch.zeros((self.rows, batch_size), device=self.dev) # store latest output from adding a batch
            self.scaler_col_act_alt = torch.zeros((self.rows), device=self.dev)
            # Initialize raw activation storage as a dummy tensor.
            self.col_activations = torch.zeros((1, self.rows), device=self.dev) # record individual activations of output for the gate projection

        if layer_name == 'mlp.up_proj': 
            self.row_activations = torch.zeros((1, self.columns), device=self.dev) # record individual activations of input for the up projection
            self.importance_scores = torch.zeros_like(layer.weight.data)

        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out, gate_activations=None):

        # Ensure input has a batch dimension.
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if tmp != self.batch_size: # added this to check
            print('tmp: ', tmp)
            print('inp.shape: ', inp.shape)
            print('self.batch_size: ', self.batch_size)
            raise ValueError('')

        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()  # Transpose so each row corresponds to a feature.

        # Update running input statistics.
        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)

        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples # taking L2-norm accross the nsamples x seq. length, input activations are tracked across calibration samples

        # Update output statistics.
        p = 1
        if isinstance(self.layer, nn.Linear):
            if len(out.shape) == 3:
                out = out.reshape((-1, out.shape[-1]))
            out = out.t()  # Now out is of shape (rows, batch_size)

        out = out.type(torch.float32)

        self.scaler_col *= (self.nsamples - tmp) / self.nsamples  # Rolling average update
        self.scaler_col += torch.norm(out, p=p, dim=1) ** 2 / self.nsamples

        # Update SiLU activated output statistics.
        if self.layer_name == 'mlp.gate_proj': 
            activated_out = F.silu(out)

            self.scaler_col_act *= (self.nsamples - tmp) / self.nsamples
            self.scaler_col_act += torch.norm(activated_out, p=p, dim=1) ** 2 / self.nsamples

            self.scaler_col_latest = activated_out

        if hasattr(self, 'importance_scores'):

            self.importance_scores += torch.einsum('ik,jk->ji', torch.abs(inp), torch.abs(gate_activations)) 