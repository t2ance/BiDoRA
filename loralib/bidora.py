import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import LoRALayer


class BiDoRALinear(nn.Linear, LoRALayer):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 8,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            fan_in_fan_out: bool = False,
            merge_weights: bool = True,
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        self.fan_in_fan_out = fan_in_fan_out
        if r > 0:
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r, in_features))
            )
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features, r))
            )
            self.ranknum = r
            self.scaling = (self.lora_alpha if self.lora_alpha > 0 else float(self.r)) / (self.ranknum + 1e-5)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.normal_(self.lora_A)
            nn.init.zeros_(self.lora_B)

    def train(self, alphas):
        nn.Linear.train(self)
        if self.merge_weights and self.merged:
            raise NotImplementedError()

    def eval(self, alphas):
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            raise NotImplementedError()

    def w_ft(self, m_ft):
        delta_v = self.lora_B @ self.lora_A
        w = self.weight + delta_v
        scaling = m_ft / w.norm(p=2, dim=0, keepdim=True)
        return w * scaling

    def w_0(self):
        return self.weight

    def v_ft(self):
        delta_v = self.lora_B @ self.lora_A
        w = self.weight + delta_v
        return w / w.norm(p=2, dim=0, keepdim=True)


    def forward(self, x: torch.Tensor, alphas):
        x = self.lora_dropout(x)

        def T(w):
            return w.T if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            mw = self.w_ft(alphas)
            result = F.linear(x, T(mw.T), bias=self.bias)
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)
