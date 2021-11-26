"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import torch
from torch import nn as nn
from torch.nn import functional as F

from bandu.imports.rlkit.policies.base import Policy
from bandu.imports.rlkit.torch import pytorch_util as ptu
from bandu.imports.rlkit.torch.core import PyTorchModule
from bandu.imports.rlkit.torch.data_management.normalizer import TorchFixedNormalizer, TorchNormalizer, CompositeNormalizer
from bandu.imports.rlkit.torch.modules import LayerNorm
from bandu.imports.rlkit.torch.relational.relational_util import fetch_preprocessing, invert_fetch_preprocessing

def identity(x):
    return x


class Mlp(PyTorchModule):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            layer_norm_kwargs=None,
    ):
        self.save_init_params(locals())
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        if isinstance(output_activation, str):
            output_activation = getattr(torch, output_activation)
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = nn.ModuleList([])
        self.layer_norms = nn.ModuleList([])
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            h = self.hidden_activation(h)
            if self.layer_norm and i < len(self.fcs):
                h = self.layer_norms[i](h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output


class FlattenMlp(Mlp):
    """
    Flatten inputs along dimension 1 and then pass through MLP.
    """

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=1)
        return super().forward(flat_inputs, **kwargs)


class CompositeNormalizedFlattenMlp(FlattenMlp):
    def __init__(
            self,
            *args,
            composite_normalizer: CompositeNormalizer = None,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        assert composite_normalizer is not None
        self.composite_normalizer = composite_normalizer

    def forward(
            self,
            observations,
            actions,
            return_preactivations=False):
        obs, _ = self.composite_normalizer.normalize_all(observations, None)
        flat_input = torch.cat((obs, actions), dim=1)
        return super().forward(flat_input, return_preactivations=return_preactivations)


class QNormalizedFlattenMlp(FlattenMlp):
    def __init__(
            self,
            *args,
            composite_normalizer: CompositeNormalizer = None,
            clip_high=float('inf'),
            clip_low=float('-inf'),
            lop_state_dim=None,
            preprocessing_kwargs=None,
            num_blocks=None,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        assert composite_normalizer is not None
        self.composite_normalizer = composite_normalizer
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.lop_state_dim = lop_state_dim
        self.preprocessing_kwargs = preprocessing_kwargs
        self.num_blocks = num_blocks

    def forward(
            self,
            observations,
            actions,
            return_preactivations=False, **kwargs):
        # if self.lop_state_dim:
        #     observations = observations.narrow(1, 0,
        #                      observations.size(1) - self.lop_state_dim)  # Chop off the final 3 dimension of gripper position
        # obs, _ = self.composite_normalizer.normalize_all(observations, None)

        # if len(obs.size()) > len(actions.size()):
        #     # Unsqueeze along block dimension
        #     actions = actions.unsqueeze(1).expand(-1, obs.size(1), -1)
        #
        # flat_input = torch.cat((obs, actions), dim=-1)
        shared_state, object_goal_state = fetch_preprocessing(observations, actions=actions, normalizer=self.composite_normalizer, **self.preprocessing_kwargs)
        flat_input = invert_fetch_preprocessing(shared_state, object_goal_state, num_blocks=self.num_blocks, **self.preprocessing_kwargs)
        assert observations.size(-1) - self.lop_state_dim + actions.size(-1) == flat_input.size(-1)
        assert len(observations.size()) == len(flat_input.size()) == 2

        if return_preactivations:
            output, preactivation = super().forward(flat_input, return_preactivations=return_preactivations)
            output = torch.clamp(output, self.clip_low, self.clip_high)
            return output, preactivation
        else:
            output = super().forward(flat_input)
            output = torch.clamp(output, self.clip_low, self.clip_high)
            return output


class VNormalizedFlattenMlp(FlattenMlp):
    def __init__(
            self,
            *args,
            composite_normalizer: CompositeNormalizer = None,
            clip_high=float('inf'),
            clip_low=float('-inf'),
            lop_state_dim=None,
            preprocessing_kwargs=None,
            num_blocks=None,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        assert composite_normalizer is not None
        self.composite_normalizer = composite_normalizer
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.lop_state_dim = lop_state_dim
        self.preprocessing_kwargs = preprocessing_kwargs
        self.num_blocks = num_blocks

    def forward(
            self,
            observations,
            return_preactivations=False, **kwargs):
        # if self.lop_state_dim:
        #     observations = observations.narrow(1, 0,
        #                      observations.size(1) - self.lop_state_dim)  # Chop off the final 3 dimension of gripper position
        # obs, _ = self.composite_normalizer.normalize_all(observations, None)
        # flat_input = obs

        shared_state, object_goal_state = fetch_preprocessing(observations,
                                                                       normalizer=self.composite_normalizer,
                                                                       **self.preprocessing_kwargs)
        flat_input = invert_fetch_preprocessing(shared_state, object_goal_state, num_blocks=self.num_blocks, **self.preprocessing_kwargs)
        assert observations.size(-1) - self.lop_state_dim == flat_input.size(-1)
        assert len(observations.size()) == len(flat_input.size()) == 2

        if return_preactivations:
            output, preactivation = super().forward(flat_input, return_preactivations=return_preactivations)
            output = torch.clamp(output, self.clip_low, self.clip_high)
            return output, preactivation
        else:
            output = super().forward(flat_input)
            output = torch.clamp(output, self.clip_low, self.clip_high)
            return output


class MlpPolicy(Mlp, Policy):
    """
    A simpler interface for creating policies.
    """

    def __init__(
            self,
            *args,
            obs_normalizer: TorchFixedNormalizer = None,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer

    def forward(self, obs, **kwargs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        return super().forward(obs, **kwargs)

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return self.eval_np(obs)


class CompositeNormalizedMlpPolicy(MlpPolicy):
    def __init__(
            self,
            *args,
            composite_normalizer: CompositeNormalizer = None,
            **kwargs
    ):
        assert composite_normalizer is not None
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.composite_normalizer = composite_normalizer

    def forward(self, obs, **kwargs):
        if self.composite_normalizer:
            obs, _ = self.composite_normalizer.normalize_all(obs, None)
            return super().forward(obs, **kwargs)


class TanhMlpPolicy(MlpPolicy):
    """
    A helper class since most policies have a tanh output activation.
    """
    def __init__(self, *args, **kwargs):
        self.save_init_params(locals())
        super().__init__(*args, output_activation=torch.tanh, **kwargs)


class ActivationLoggingWrapper:
    """
    Logs activations to a list
    """
