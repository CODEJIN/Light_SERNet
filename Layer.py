import torch
from LConv import LightweightConv1d as LConv1d
from typing import Optional


class LConv_Block(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        num_head: int,
        weight_dropout_rate: float= 0.1,
        feedforward_dropout_rate: float= 0.1,
        layer_norm_eps: float= 1e-3
        ) -> None:
        super().__init__()
        
        self.lconv = torch.nn.Sequential(
            Conv1d(
                in_channels= channels,
                out_channels= channels * 2,
                kernel_size= 1,
                w_init_gain= 'glu'
                ),
            torch.nn.GLU(dim= 1),
            LConv1d(
                input_size= channels,
                kernel_size= kernel_size,
                padding= (kernel_size - 1) // 2,
                num_heads= num_head,
                weight_softmax= True,
                bias= True,
                weight_dropout= weight_dropout_rate,
                w_init_gain= 'linear'
                )
            )
        
        self.ffn = torch.nn.Sequential(
            Conv1d(
                in_channels= channels,
                out_channels= channels * 4,
                kernel_size= 1,
                w_init_gain= 'relu'
                ),
            torch.nn.ReLU(),
            torch.nn.Dropout(p= feedforward_dropout_rate),
            Conv1d(
                in_channels= channels * 4,
                out_channels= channels,
                kernel_size= 1,
                w_init_gain= 'linear'
                ),
            Lambda(lambda x: x.permute(0, 2, 1)),
            torch.nn.LayerNorm(channels, eps= layer_norm_eps),
            Lambda(lambda x: x.permute(0, 2, 1))
            )
        
    def forward(
        self,
        x: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        '''
        x: [Batch, Dim, Time]
        '''
        x = self.lconv(x) + x
        x = self.ffn(x) + x

        if not masks is None:
            x = x.masked_fill(masks.unsqueeze(1), 0.0)

        return x

class Conv1d(torch.nn.Conv1d):
    def __init__(self, w_init_gain= 'linear', *args, **kwargs):
        self.w_init_gain = w_init_gain
        super().__init__(*args, **kwargs)

    def reset_parameters(self):
        if self.w_init_gain in ['relu', 'leaky_relu']:
            torch.nn.init.kaiming_uniform_(self.weight, nonlinearity= self.w_init_gain)
        elif self.w_init_gain == 'glu':
            assert self.out_channels % 2 == 0, 'The out_channels of GLU requires even number.'
            torch.nn.init.kaiming_uniform_(self.weight[:self.out_channels // 2], nonlinearity= 'linear')
            torch.nn.init.xavier_uniform_(self.weight[self.out_channels // 2:], gain= torch.nn.init.calculate_gain('sigmoid'))
        else:
            torch.nn.init.xavier_uniform_(self.weight, gain= torch.nn.init.calculate_gain(self.w_init_gain))
        if not self.bias is None:
            torch.nn.init.zeros_(self.bias)

class Conv2d(torch.nn.Conv2d):
    def __init__(self, w_init_gain= 'linear', *args, **kwargs):
        self.w_init_gain = w_init_gain
        super().__init__(*args, **kwargs)

    def reset_parameters(self):
        if self.w_init_gain in ['relu', 'leaky_relu']:
            torch.nn.init.kaiming_uniform_(self.weight, nonlinearity= self.w_init_gain)
        elif self.w_init_gain == 'glu':
            assert self.out_channels % 2 == 0, 'The out_channels of GLU requires even number.'
            torch.nn.init.kaiming_uniform_(self.weight[:self.out_channels // 2], nonlinearity= 'linear')
            torch.nn.init.xavier_uniform_(self.weight[self.out_channels // 2:], gain= torch.nn.init.calculate_gain('sigmoid'))
        else:
            torch.nn.init.xavier_uniform_(self.weight, gain= torch.nn.init.calculate_gain(self.w_init_gain))
        if not self.bias is None:
            torch.nn.init.zeros_(self.bias)

class Linear(torch.nn.Linear):
    def __init__(self, w_init_gain= 'linear', *args, **kwargs):
        self.w_init_gain = w_init_gain
        super().__init__(*args, **kwargs)

    def reset_parameters(self):
        if self.w_init_gain in ['relu', 'leaky_relu']:
            torch.nn.init.kaiming_uniform_(self.weight, nonlinearity= self.w_init_gain)
        elif self.w_init_gain == 'glu':
            assert self.out_channels % 2 == 0, 'The out_channels of GLU requires even number.'
            torch.nn.init.kaiming_uniform_(self.weight[:self.out_channels // 2], nonlinearity= 'linear')
            torch.nn.init.xavier_uniform_(self.weight[self.out_channels // 2:], gain= torch.nn.init.calculate_gain('sigmoid'))
        else:
            torch.nn.init.xavier_uniform_(self.weight, gain= torch.nn.init.calculate_gain(self.w_init_gain))
        if not self.bias is None:
            torch.nn.init.zeros_(self.bias)

class AdaIN1d(torch.nn.InstanceNorm1d):
    def __init__(self, condition_features: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.condition = Linear(
            in_features= condition_features,
            out_features= self.num_features * 2
            )

    def forward(
        self,
        x: torch.Tensor,
        conditions: torch.Tensor
        ):
        x = super().forward(x)
        betas, gammas = self.condition(conditions).unsqueeze(2).chunk(chunks= 2, dim= 1)
        x = (gammas + 1) * x + betas

        return x

class AdaIN2d(torch.nn.InstanceNorm2d):
    def __init__(self, condition_features: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.condition = Linear(
            in_features= condition_features,
            out_features= self.num_features * 2
            )

    def forward(
        self,
        x: torch.Tensor,
        conditions: torch.Tensor
        ):
        x = super().forward(x)
        betas, gammas = self.condition(conditions).unsqueeze(2).unsqueeze(2).chunk(chunks= 2, dim= 1)
        x = (gammas + 1) * x + betas

        return x


class Lambda(torch.nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)