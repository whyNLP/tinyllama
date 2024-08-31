import torch
import torch.nn as nn
from xformers.ops.swiglu_op import swiglu, unbind

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        # use a single weight matrix to store the weights of gate_proj and up_proj
        w1w2 = nn.Parameter(torch.empty([2, self.intermediate_size, self.hidden_size]), requires_grad=False)
        w1, w2 = unbind(w1w2, dim=0)
        # initialize the weights
        w1.data = self.gate_proj.weight.data
        w2.data = self.up_proj.weight.data
        # bind the weights to the linear layers
        self.gate_proj.weight = nn.Parameter(w1)
        self.up_proj.weight = nn.Parameter(w2)

        assert config.hidden_act == 'silu', f"Fused SwiGLU requires silu as activate function, but {config.hidden_act} found."
        if hasattr(config, 'pretraining_tp'):
            assert config.pretraining_tp == 1, "Fused SwiGLU requires pretraining_tp == 1"

    def forward(self, x):
        return swiglu(
            x,
            self.gate_proj.weight,
            self.gate_proj.bias,
            self.up_proj.weight,
            self.up_proj.bias,
            self.down_proj.weight,
            self.down_proj.bias,
            op=None
        )
