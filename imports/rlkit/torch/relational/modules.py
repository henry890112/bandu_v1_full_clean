import math

import torch.nn as nn
from torch import nn as nn
from torch.nn import Parameter, functional as F

from bandu.imports.rlkit.torch.core import PyTorchModule
from bandu.imports.rlkit.torch.networks import Mlp
import torch

from bandu.imports.rlkit.torch.relational.relational_util import fetch_preprocessing

import bandu.imports.rlkit.torch.pytorch_util as ptu


class FetchInputPreprocessing(PyTorchModule):
    """
    Used for the Q-value and value function

    Takes in either obs or (obs, actions) in the forward function and returns the same sized embedding for both

    Make sure actions are being passed in!!
    """
    def __init__(self,
                 normalizer,
                 object_total_dim,
                 embedding_dim,
                 layer_norm=True):
        self.save_init_params(locals())
        super().__init__()
        self.normalizer = normalizer
        self.fc_embed = nn.Linear(object_total_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim) if layer_norm else None

    def forward(self, obs, actions=None, mask=None):
        vertices = fetch_preprocessing(obs, actions=actions, normalizer=self.normalizer, mask=mask)

        if self.layer_norm is not None:
            return self.layer_norm(self.fc_embed(vertices))
        else:
            return self.fc_embed(vertices)


class Attention(PyTorchModule):
    """
    Additive, multi-headed attention
    """
    def __init__(self,
                 embedding_dim,
                 num_heads=1,
                 layer_norm=True,
                 activation_fnx=F.leaky_relu,
                 query_key_activation_fnx=None,
                 softmax_temperature=1.0):
        assert query_key_activation_fnx is not None
        self.save_init_params(locals())
        super().__init__()
        self.fc_createheads = nn.Linear(embedding_dim, num_heads * embedding_dim)
        self.fc_logit = nn.Linear(embedding_dim, 1)
        self.fc_reduceheads = nn.Linear(num_heads * embedding_dim, embedding_dim)
        self.query_norm = nn.LayerNorm(embedding_dim) if layer_norm else None
        # self.layer_norms = nn.ModuleList([nn.LayerNorm(i) for i in [num_heads*embedding_dim, 1, embedding_dim]]) if layer_norm else None
        self.softmax_temperature = Parameter(torch.tensor(softmax_temperature))
        self.query_key_activation_fnx = query_key_activation_fnx
        self.activation_fnx = activation_fnx

    def forward(self, query, context, memory, mask, reduce_heads=True, return_probs=False):
        """
        N, nV, nE memory -> N, nV, nE updated memory

        :param query:
        :param context:
        :param memory:
        :param mask: N, nV
        :return:
        """
        N, nQ, nE = query.size()
        # assert len(query.size()) == 3

        # assert self.fc_createheads.out_features % nE == 0
        nH = int(self.fc_createheads.out_features / nE)

        nV = memory.size(1)

        # assert len(mask.size()) == 2

        # N, nQ, nE -> N, nQ, nH, nE
        # if nH > 1:
        query = self.fc_createheads(query).view(N, nQ, nH, nE)
        # else:
        #     query = query.view(N, nQ, nH, nE)

        if self.query_norm is not None:
            query = self.query_norm(query)
        # if self.layer_norms is not None:
        # query = self.layer_norms[0](query)
        # N, nQ, nH, nE -> N, nQ, nV, nH, nE
        query = query.unsqueeze(2).expand(-1, -1, nV, -1, -1)

        # N, nV, nE -> N, nQ, nV, nH, nE
        context = context.unsqueeze(1).unsqueeze(3).expand_as(query)

        # -> N, nQ, nV, nH, 1
        # qc_logits = self.fc_logit(torch.tanh(context + query))
        qc_logits = self.fc_logit(self.query_key_activation_fnx(context + query))

        # if self.layer_norms is not None:
        #     qc_logits = self.layer_norms[1](qc_logits)

        # N, nV -> N, nQ, nV, nH, 1
        logit_mask = mask.unsqueeze(1).unsqueeze(3).unsqueeze(-1).expand_as(qc_logits)

        # qc_logits N, nQ, nV, nH, 1 -> N, nQ, nV, nH, 1
        attention_probs = F.softmax(qc_logits / self.softmax_temperature * logit_mask + (-99999) * (1 - logit_mask), dim=2)

        if return_probs:
            ret_attention_probs = attention_probs.squeeze(-1)

        # N, nV, nE -> N, nQ, nV, nH, nE
        memory = memory.unsqueeze(1).unsqueeze(3).expand(-1, nQ, -1, nH, -1)

        # N, nV -> N, nQ, nV, nH, nE
        memory_mask = mask.unsqueeze(1).unsqueeze(3).unsqueeze(4).expand_as(memory)

        # assert memory.size() == attention_probs.size() == mask.size(), (memory.size(), attention_probs.size(), memory_mask.size())

        # N, nQ, nV, nH, nE -> N, nQ, nH, nE
        attention_heads = (memory * attention_probs * memory_mask).sum(2).squeeze(2)

        if not reduce_heads:
            return [attention_heads]

        attention_heads = self.activation_fnx(attention_heads)
        # N, nQ, nH, nE -> N, nQ, nE
        # if nQ > 1:
        attention_result = self.fc_reduceheads(attention_heads.view(N, nQ, nH*nE))
        # else:
        #     attention_result = attention_heads.view(N, nQ, nE)

        # attention_result = self.activation_fnx(attention_result)
        #TODO: add nonlinearity here...

        # if self.layer_norms is not None:
        #     attention_result = self.layer_norms[2](attention_result)

        # assert len(attention_result.size()) == 3

        if return_probs:
            return [attention_result, ptu.get_numpy(ret_attention_probs)]
        else:
            return [attention_result]


class MultiobjectAttention(PyTorchModule):
    """
    Additive, multi-headed attention with addition batch dimension for multiple objects
    """
    def __init__(self,
                 embedding_dim,
                 num_heads=1,
                 layer_norm=True,
                 activation_fnx=F.leaky_relu,
                 query_key_activation_fnx=None,
                 softmax_temperature=1.0):
        assert query_key_activation_fnx is not None
        self.save_init_params(locals())
        super().__init__()
        self.fc_createheads = nn.Linear(embedding_dim, num_heads * embedding_dim)
        self.fc_logit = nn.Linear(embedding_dim, 1)
        self.fc_reduceheads = nn.Linear(num_heads * embedding_dim, embedding_dim)
        self.query_norm = nn.LayerNorm(embedding_dim) if layer_norm else None
        # self.layer_norms = nn.ModuleList([nn.LayerNorm(i) for i in [num_heads*embedding_dim, 1, embedding_dim]]) if layer_norm else None
        self.softmax_temperature = Parameter(torch.tensor(softmax_temperature))
        self.query_key_activation_fnx = query_key_activation_fnx
        self.activation_fnx = activation_fnx

    def forward(self, query, context, memory, mask, reduce_heads=True, return_probs=False):
        """
        N, nO, nV, nE memory -> N, nO, nV, nE updated memory

        :param query:
        :param context:
        :param memory:
        :param mask: N, nV
        :return:
        """
        N, nO, nQ, nE = query.size()
        nH = int(self.fc_createheads.out_features / nE)

        # memory: N, nO, nV, nE
        nV = memory.size(-2)

        # N, nO, nQ, nE -> N, nO, nQ, nH, nE
        # if nH > 1:
        query = self.fc_createheads(query).view(N, nO, nQ, nH, nE)
        # else:
        #     query = query.view(N, nQ, nH, nE)

        if self.query_norm is not None:
            query = self.query_norm(query)
        # if self.layer_norms is not None:
        # query = self.layer_norms[0](query)
        # N, nO, nQ, nH, nE -> N,  nO, nQ, nV, nH, nE

        # print("query shape")
        # print(query.shape)
        query = query.unsqueeze(-3).expand(-1, -1, -1, nV, -1, -1)

        # N, nO, nV, nE -> N, nO, nQ, nV, nH, nE
        # print("qc shapes")
        # print(context.shape)
        # print(query.shape)
        context = context.unsqueeze(2).unsqueeze(4).expand_as(query)

        # -> N, nO, nQ, nV, nH, 1
        # qc_logits = self.fc_logit(torch.tanh(context + query))
        qc_logits = self.fc_logit(self.query_key_activation_fnx(context + query))

        # if self.layer_norms is not None:
        #     qc_logits = self.layer_norms[1](qc_logits)

        # N, nO, nV -> N, nO, nQ, nV, nH, 1
        logit_mask = mask.unsqueeze(2).unsqueeze(4).unsqueeze(-1).expand_as(qc_logits)

        # qc_logits: N, nO, nQ, nV, nH, 1 -> N, nO, nQ, nV, nH, 1
        attention_probs = F.softmax(qc_logits / self.softmax_temperature * logit_mask + (-99999) * (1 - logit_mask), dim=-3)

        if return_probs:
            ret_attention_probs = attention_probs.squeeze(-1)

        # N, nO, nV, nE -> N, nO, nQ, nV, nH, nE
        memory = memory.unsqueeze(2).unsqueeze(4).expand(-1, -1, nQ, -1, nH, -1)

        # N, nO, nV -> N, nO, nQ, nV, nH, nE
        memory_mask = mask.unsqueeze(2).unsqueeze(4).unsqueeze(5).expand_as(memory)

        # assert memory.size() == attention_probs.size() == mask.size(), (memory.size(), attention_probs.size(), memory_mask.size())

        # N, nO, nQ, nV, nH, nE -> N, nO, nQ, nH, nE
        attention_heads = (memory * attention_probs * memory_mask).sum(-3).squeeze(-3)

        if not reduce_heads:
            return [attention_heads]

        attention_heads = self.activation_fnx(attention_heads)
        # N, nO, nQ, nH, nE -> N, nO, nQ, nE
        # if nQ > 1:
        attention_result = self.fc_reduceheads(attention_heads.view(N, nO, nQ, nH*nE))
        # else:
        #     attention_result = attention_heads.view(N, nQ, nE)

        # attention_result = self.activation_fnx(attention_result)
        #TODO: add nonlinearity here...

        # if self.layer_norms is not None:
        #     attention_result = self.layer_norms[2](attention_result)

        # assert len(attention_result.size()) == 3

        if return_probs:
            return [attention_result, ptu.get_numpy(ret_attention_probs)]
        else:
            return [attention_result]


class AttentiveGraphToGraph(PyTorchModule):
    """
    Uses attention to perform message passing between 1-hop neighbors in a fully-connected graph
    """
    def __init__(self,
                 embedding_dim=64,
                 num_heads=1,
                 layer_norm=True,
                 attention_kwargs=None,
                 num_qcm_modules=1,
                 multiobject_attention=False,
                 **kwargs):
        def lin_module_gen(embedding_dim):
            return nn.Sequential(nn.Linear(embedding_dim,embedding_dim),
                                 nn.LeakyReLU(),
                                 nn.LayerNorm(embedding_dim))
        if attention_kwargs is None:
            attention_kwargs = dict()
        self.save_init_params(locals())
        super().__init__()

        self.vertex_mlp_fc_list = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim) for i in range(num_qcm_modules)])
        self.vertex_mlp_norm_list = nn.ModuleList([nn.LayerNorm(embedding_dim) if layer_norm else None for i in range(num_qcm_modules)])
        self.fc_qcm = nn.Linear(embedding_dim, 3 * embedding_dim)
        # self.fc_qcm = nn.Sequential(*[lin_module_gen(embedding_dim) for i in range(num_qcm_modules)],
        #                             nn.Linear(embedding_dim, 3 * embedding_dim))
        if multiobject_attention:
            self.attention = MultiobjectAttention(embedding_dim, num_heads=num_heads, layer_norm=layer_norm, **attention_kwargs)
        else:
            self.attention = Attention(embedding_dim, num_heads=num_heads, layer_norm=layer_norm, **attention_kwargs)
        # self.layer_norm= nn.LayerNorm(3*embedding_dim) if layer_norm else None

    def forward(self, vertices, mask, **kwargs):
        """

        :param vertices: N x nV x nE
        :return: updated vertices: N x nV x nE
        """
        # assert len(vertices.size()) == 3
        # N, nV, nE = vertices.size()
        # assert mask.size() == torch.Size([N, nV]), (mask.size(), torch.Size([N, nV]))

        # -> (N, nQ, nE), (N, nV, nE), (N, nV, nE)

        # if self.layer_norm is not None:
        #     qcm_block = self.layer_norm(self.fc_qcm(vertices))
        # else:

        for i in range(len(self.vertex_mlp_fc_list)):
            vertices = F.leaky_relu(self.vertex_mlp_fc_list[i](vertices) + vertices)
            if self.vertex_mlp_norm_list[i] is not None:
                vertices = self.vertex_mlp_norm_list[i](vertices)
        qcm_block = self.fc_qcm(vertices)

        query, context, memory = qcm_block.chunk(3, dim=-1)

        return self.attention(query, context, memory, mask, **kwargs)


class AttentiveGraphPooling(PyTorchModule):
    """
    Pools nV vertices to a single vertex embedding

    """
    def __init__(self,
                 embedding_dim=64,
                 num_heads=1,
                 init_w=3e-3,
                 layer_norm=True,
                 mlp_kwargs=None,
                 attention_kwargs=None,
                 multiobject_attention=False):
        # assert num_objects is not None, "You must pass in num_objects"
        self.save_init_params(locals())
        super().__init__()
        self.fc_cm = nn.Linear(embedding_dim, 2 * embedding_dim)
        self.layer_norm = nn.LayerNorm(2*embedding_dim) if layer_norm else None

        self.input_independent_query = Parameter(torch.Tensor(embedding_dim))
        self.input_independent_query.data.uniform_(-init_w, init_w)
        # self.num_heads = num_heads
        if multiobject_attention:
            self.attention = MultiobjectAttention(embedding_dim, num_heads=num_heads, layer_norm=layer_norm, **attention_kwargs)
        else:
            self.attention = Attention(embedding_dim, num_heads=num_heads, layer_norm=layer_norm, **attention_kwargs)

        if mlp_kwargs is not None:
            self.proj = Mlp(**mlp_kwargs)
        else:
            self.proj = None
        self.multiobject_attention = multiobject_attention
        # self.num_objects = num_objects

    def forward(self, vertices, mask, **kwargs):
        """
        N, nO, nV, nE -> N, nO, nE
        :param vertices:
        :param mask:
        :return: list[attention_result]
        """
        if self.multiobject_attention:
            N, nO, nV, nE = vertices.size()
            # nE -> N, nO, nQ, nE where nQ == self.num_heads
            nQ = 1
            query = self.input_independent_query.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(N, nO, nQ, -1)
        else:
            N, nV, nE = vertices.size()
            # nE -> N, nQ, nE where nQ == self.num_heads
            query = self.input_independent_query.unsqueeze(0).unsqueeze(0).expand(N, 1, -1)


        if self.layer_norm is not None:
            cm_block = self.layer_norm(self.fc_cm(vertices))
        else:
            cm_block = self.fc_cm(vertices)
        context, memory = cm_block.chunk(2, dim=-1)
        # context = vertices
        # memory = vertices

        # gt.stamp("Readout_preattention")
        attention_out = self.attention(query, context, memory, mask, **kwargs)

        attention_result = attention_out[0]
        if 'reduce_heads' in kwargs and not kwargs['reduce_heads']:
            assert len(attention_result.shape) == 4
            return [attention_result]

        # gt.stamp("Readout_postattention")
        # return attention_result.sum(dim=1) # Squeeze nV dimension so that subsequent projection function does not have a useless 1 dimension

        if self.multiobject_attention:
            squeeze_idx = 2
        else:
            squeeze_idx = 1

        if self.proj is not None:
            ret_out = [self.proj(attention_result).squeeze(squeeze_idx)]
        else:
            ret_out = [attention_result.squeeze(squeeze_idx)]

        if 'return_probs' in kwargs and kwargs['return_probs']:
            ret_out.append(attention_out[1])
        return ret_out
        # else:
        #     return attention_result