import torch
from torch import nn
from transformers import BertConfig, BertModel
import numpy as np
from utils.helper import masked_operation
from torch.autograd import Function


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx, input, coeff=1.):
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)


class WarmStartGradientReverseLayer(nn.Module):
    """Gradient Reverse Layer :math:`\mathcal{R}(x)` with warm start
        The forward and backward behaviours are:
        .. math::
            \mathcal{R}(x) = x,
            \dfrac{ d\mathcal{R}} {dx} = - \lambda I.
        :math:`\lambda` is initiated at :math:`lo` and is gradually changed to :math:`hi` using the following schedule:
        .. math::
            \lambda = \dfrac{2(hi-lo)}{1+\exp(- α \dfrac{i}{N})} - (hi-lo) + lo
        where :math:`i` is the iteration step.
        Args:
            alpha (float, optional): :math:`α`. Default: 1.0
            lo (float, optional): Initial value of :math:`\lambda`. Default: 0.0
            hi (float, optional): Final value of :math:`\lambda`. Default: 1.0
            max_iters (int, optional): :math:`N`. Default: 1000
            auto_step (bool, optional): If True, increase :math:`i` each time `forward` is called.
              Otherwise use function `step` to increase :math:`i`. Default: False
        """

    def __init__(self, alpha=1.0, lo=0.0, hi=1., max_iters=1000., auto_step=False):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input):
        """"""
        coeff = np.float(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        """Increase iteration number :math:`i` by 1"""
        self.iter_num += 1


class SequenceDiscriminator(nn.Module):
    def __init__(self, dim, layer_num=3, head_num=8, p=0.6):
        super().__init__()
        self.encoder_cfg = BertConfig(hidden_size=dim, num_hidden_layers=layer_num, num_attention_heads=head_num,
                                      attention_probs_dropout_prob=p, hidden_dropout_prob=p)
        self.encoder = BertModel(self.encoder_cfg)
        self.disc_linear = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Dropout(p),
            nn.Linear(dim, dim), nn.GELU(), nn.Dropout(p),
            nn.Linear(dim, 1)
        )

    def forward(self, inputs, mask=None):
        # encoding: (B, L, D), mask: (B, L)
        inputs = self.encoder(input_ids=inputs, token_type_ids=None, attention_mask=mask,
                              output_hidden_states=True).last_hidden_state  # (B, L, D)
        if mask is None:
            inputs = inputs.mean(1)
        else:
            inputs = masked_operation(inputs, mask, dim=1, operation="mean")
        return self.disc_linear(inputs).squeeze(-1)


class PretrainedDiscriminator(nn.Module):
    def __init__(self, dim=768, p=0.6, **kwargs):
        super().__init__()
        self.encoder = BertModel.from_pretrained("")
        self.disc_linear = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Dropout(p),
            nn.Linear(dim, dim), nn.GELU(), nn.Dropout(p),
            nn.Linear(dim, 1)
        )

    def forward(self, inputs, mask=None):
        # encoding: (B, L, D), mask: (B, L)
        inputs = self.encoder(input_ids=inputs, token_type_ids=None, attention_mask=mask,
                              output_hidden_states=True).last_hidden_state.detach()  # (B, L, D)
        if mask is None:
            inputs = inputs.mean(1)
        else:
            inputs = masked_operation(inputs, mask, dim=1, operation="mean")
        return self.disc_linear(inputs).squeeze(-1)

class SingleDiscriminator(nn.Module):
    def __init__(self, dim, need_grl=True):
        super().__init__()
        # self.grl = WarmStartGradientReverseLayer(alpha=2., lo=0., hi=1., max_iters=1000, auto_step=True)
        self.grl = GradientReverseLayer() if need_grl else None
        self.disc_linear = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(inplace=True),
            nn.Linear(dim, dim), nn.ReLU(inplace=True),
            nn.Linear(dim, 1)
        )

    def forward(self, encoding):
        if self.grl is not None:
            inputs = self.grl(encoding)
            return self.disc_linear(inputs).squeeze(-1)
        return self.disc_linear(encoding).squeeze(-1)
