import torch.nn as nn
import torch
from loralib.bidora import BiDoRALinear


class BiDoRAArchitecture(torch.nn.Module):
    """
    #parameter is 1 x k for each linear layer
    """

    def __init__(self, m, layer_num):
        super(BiDoRAArchitecture, self).__init__()
        magnitudes_query = []
        magnitudes_value = []
        for module_name, module in m.named_modules():
            if isinstance(module, BiDoRALinear):
                magnitude = module.weight.norm(p=2, dim=0, keepdim=False)
                print(f'Initialize from module {module_name},'
                      f' module shape: {module.weight.shape},'
                      f' magnitude shape: {magnitude.shape}')
                if module_name.endswith('.query'):
                    magnitudes_query.append(magnitude)
                elif module_name.endswith('.value'):
                    magnitudes_value.append(magnitude)
                else:
                    raise NotImplementedError()
        assert len(magnitudes_query) == len(magnitudes_value)
        assert len(magnitudes_query) == layer_num, f'{len(magnitudes_query)} {layer_num}'
        self.alpha = nn.ModuleList()
        for i, (query, value) in enumerate(zip(magnitudes_query, magnitudes_value)):
            q_v_pair = nn.ParameterList([nn.Parameter(query), nn.Parameter(value)])
            self.alpha.append(q_v_pair)

    def forward(self):
        return self.alpha
