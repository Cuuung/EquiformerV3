import torch
from functools import partial


_NORM_TYPE_LIST = [
    'equivariant_layer_norm',
    'sep_layer_norm',
    'merge_layer_norm',
    'merge_layer_norm_attn_rms_norm',   # Use `EquivariantMergeLayerNorm` for the pre-norm layer 
                                        # and `RMSNorm` for attention re-normalization
    'merge_rms_norm'
]


def get_normalization_layer(norm_type, lmax, num_channels, eps=1e-5, affine=True, normalization='component'):
    assert norm_type in _NORM_TYPE_LIST
    if norm_type == 'equivariant_layer_norm':
        norm_class = EquivariantLayerNorm
    elif norm_type == 'sep_layer_norm':
        norm_class = EquivariantSeparableLayerNorm
    elif norm_type in ['merge_layer_norm', 'merge_layer_norm_attn_rms_norm']:
        norm_class = EquivariantMergeLayerNorm
    elif norm_type == 'merge_rms_norm':
        norm_class = partial(EquivariantMergeLayerNorm, centering=False)
    else:
        raise ValueError
    return norm_class(lmax, num_channels, eps, affine, normalization)


class EquivariantLayerNorm(torch.nn.Module):
    def __init__(self, lmax, num_channels, eps=1e-5, affine=True, normalization='component'):
        super().__init__()
        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        
        if affine:
            self.affine_weight = torch.nn.Parameter(torch.ones((self.lmax + 1), self.num_channels))
            self.affine_bias   = torch.nn.Parameter(torch.zeros(self.num_channels))
        else:
            self.register_parameter('affine_weight', None)
            self.register_parameter('affine_bias', None)

        assert normalization in ['norm', 'component']
        self.normalization = normalization


    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, eps={self.eps})"


    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, inputs):
        """
            1.   `inputs` shape: (num_nodes, (self.lmax + 1) ** 2, self.num_channels)
        """
        outputs = []
        
        for l in range(self.lmax + 1):
            start_idx = l ** 2
            length = 2 * l + 1
            
            feature = inputs.narrow(1, start_idx, length)
            
            # For scalars, first compute and subtract the mean
            if l == 0:
                feature_mean = torch.mean(feature, dim=2, keepdim=True)
                feature = feature - feature_mean
                
            # Then compute the rescaling factor (norm of each feature vector)
            # Rescaling of the norms themselves based on the option "normalization"
            if self.normalization == 'norm':
                feature_norm = feature.pow(2).sum(dim=1, keepdim=True)      # [N, 1, C]
            elif self.normalization == 'component':
                feature_norm = feature.pow(2).mean(dim=1, keepdim=True)     # [N, 1, C]
            
            feature_norm = torch.mean(feature_norm, dim=2, keepdim=True)    # [N, 1, 1]
            feature_norm = (feature_norm + self.eps).pow(-0.5)
            
            if self.affine:
                weight = self.affine_weight.narrow(0, l, 1)     # [1, C]
                weight = weight.view(1, 1, -1)                  # [1, 1, C]
                feature_norm = feature_norm * weight            # [N, 1, C]
            
            feature = feature * feature_norm
            
            if self.affine and l == 0: 
                bias = self.affine_bias
                bias = bias.view(1, 1, -1)
                feature = feature + bias
            
            outputs.append(feature)
        
        outputs = torch.cat(outputs, dim=1)
        
        return outputs
        

class EquivariantSeparableLayerNorm(torch.nn.Module):
    """
        1.  Use `expand_index` to skip for loop during affine transformation.
    """
    def __init__(self, lmax, num_channels, eps=1e-5, affine=True, normalization='component', std_balance_degrees=True):
        super().__init__()
        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.std_balance_degrees = std_balance_degrees

        # for L = 0
        self.norm_l0 = torch.nn.LayerNorm(self.num_channels, eps=self.eps, elementwise_affine=self.affine)

        # for L > 0
        if self.affine:
            self.affine_weight = torch.nn.Parameter(torch.ones(self.lmax, self.num_channels))
            expand_index = torch.zeros([((self.lmax + 1) ** 2 - 1)]).long()     # L > 0
            for l in range(1, self.lmax + 1):
                start_idx = l ** 2 - 1
                length = 2 * l + 1
                expand_index[start_idx : (start_idx + length)] = (l - 1)
            self.register_buffer('expand_index', expand_index)
        else:
            self.register_parameter('affine_weight', None)

        assert normalization in ['norm', 'component']
        self.normalization = normalization

        if self.std_balance_degrees:
            balance_degree_weight = torch.zeros((self.lmax + 1) ** 2 - 1, 1)
            for l in range(1, self.lmax + 1):
                start_idx = l ** 2 - 1
                length = 2 * l + 1
                balance_degree_weight[start_idx : (start_idx + length), :] = (1.0 / length)
            balance_degree_weight = balance_degree_weight / self.lmax
            balance_degree_weight = balance_degree_weight.permute((1, 0))
            self.register_buffer('balance_degree_weight', balance_degree_weight)
        else:
            self.balance_degree_weight = None

    
    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, eps={self.eps}, std_balance_degrees={self.std_balance_degrees})"


    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, inputs):
        """
            1.  `inputs` shape: (num_nodes, (self.lmax + 1) ** 2, self.num_channels)
        """
        outputs = []

        # for L = 0
        scalars = inputs.narrow(1, 0, 1)
        scalars = self.norm_l0(scalars)
        outputs.append(scalars)

        # for L > 0
        if self.lmax > 0:
            num_m_components = (self.lmax + 1) ** 2
            feature = inputs.narrow(1, 1, num_m_components - 1)

            feature_norm = feature.pow(2)
            feature_norm = torch.mean(feature_norm, dim=2, keepdim=True)        # [N, (L_max + 1)**2 - 1, 1]
            
            # Then compute the rescaling factor (norm of each feature vector)
            # Rescaling of the norms themselves based on the option "normalization"
            if self.normalization == 'norm':
                feature_norm = feature_norm.sum(dim=1, keepdim=True)            # [N, 1, 1]
            elif self.normalization == 'component':
                if self.std_balance_degrees:
                    #feature_norm = feature.pow(2)                               # [N, (L_max + 1)**2 - 1, C], without L = 0
                    #feature_norm = torch.einsum('nic, ia -> nac', feature_norm, self.balance_degree_weight) # [N, 1, C]
                    feature_norm = torch.einsum('ai, nic -> nac', self.balance_degree_weight, feature_norm) # [N, 1, C]
                    #feature_norm = torch.matmul(self.balance_degree_weight, feature_norm) # [N, 1, 1]
                else:
                    feature_norm = feature_norm.mean(dim=1, keepdim=True)       # [N, 1, 1]

            feature_norm = (feature_norm + self.eps).pow(-0.5)

            if self.affine:
                weight = self.affine_weight.view(1, self.lmax, self.num_channels)
                weight = torch.index_select(weight, dim=1, index=self.expand_index)
                feature_norm = feature_norm * weight
            feature = feature * feature_norm

            outputs.append(feature)

        outputs = torch.cat(outputs, dim=1)
        return outputs


class EquivariantMergeLayerNorm(torch.nn.Module):
    """
        1.  Use `expand_index` to skip for loop during affine transformation.
        2.  Different from `EquivariantSeparableLayerNorm`, we normalize over all degrees L >= 0.
        3.  If `centering == False`, this becomes RMSNorm for all degrees.
    """
    def __init__(self, lmax, num_channels, eps=1e-5, affine=True, normalization='component', std_balance_degrees=True, centering=True):
        super().__init__()
        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.std_balance_degrees = std_balance_degrees
        self.centering = centering

        if self.affine:
            self.affine_weight = torch.nn.Parameter(torch.ones((self.lmax + 1), self.num_channels))
            expand_index = torch.zeros([((self.lmax + 1) ** 2)]).long()     # L >= 0
            for l in range(self.lmax + 1):
                start_idx = l ** 2
                length = 2 * l + 1
                expand_index[start_idx : (start_idx + length)] = l
            self.register_buffer('expand_index', expand_index)

            if self.centering:
                self.affine_bias = torch.nn.Parameter(torch.zeros(self.num_channels))
            else:
                self.register_parameter('affine_bias', None)
        else:
            self.register_parameter('affine_weight', None)
            self.register_parameter('affine_bias', None)

        assert normalization in ['norm', 'component']
        self.normalization = normalization

        if self.std_balance_degrees:
            balance_degree_weight = torch.zeros((self.lmax + 1) ** 2, 1)
            for l in range(self.lmax + 1):
                start_idx = l ** 2
                length = 2 * l + 1
                balance_degree_weight[start_idx : (start_idx + length), :] = (1.0 / length)
            balance_degree_weight = balance_degree_weight / (self.lmax + 1)
            balance_degree_weight = balance_degree_weight.permute((1, 0))
            self.register_buffer('balance_degree_weight', balance_degree_weight)
        else:
            self.balance_degree_weight = None

    
    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, eps={self.eps}, std_balance_degrees={self.std_balance_degrees}, centering={self.centering})"


    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, inputs):
        """
            1.  `inputs` shape: (num_nodes, (self.lmax + 1) ** 2, self.num_channels)
        """
        # for L = 0
        if self.centering:
            scalars = inputs.narrow(1, 0, 1)
            scalars_mean = scalars.mean(dim=2, keepdim=True) # [N, 1, 1]
            scalars = scalars - scalars_mean
            inputs = torch.cat((scalars, inputs.narrow(1, 1, inputs.shape[1] - 1)), dim=1)

        # for L >= 0
        feature_norm = inputs.pow(2)
        feature_norm = torch.mean(feature_norm, dim=2, keepdim=True)        # [N, (L_max + 1)**2, 1]
        if self.normalization == 'norm':
            feature_norm = feature_norm.sum(dim=1, keepdim=True)            # [N, 1, 1]
        elif self.normalization == 'component':
            if self.std_balance_degrees:
                feature_norm = torch.einsum('ai, nic -> nac', self.balance_degree_weight, feature_norm) # [N, 1, 1]
            else:
                feature_norm = feature_norm.mean(dim=1, keepdim=True)       # [N, 1, 1]
        feature_norm = (feature_norm + self.eps).pow(-0.5)
        if self.affine:
            weight = self.affine_weight.view(1, (self.lmax + 1), self.num_channels)
            weight = torch.index_select(weight, dim=1, index=self.expand_index)
            feature_norm = feature_norm * weight
        outputs = inputs * feature_norm

        if self.affine and self.centering:
            outputs[:, 0:1, :] = outputs.narrow(1, 0, 1) + self.affine_bias.view(1, 1, self.num_channels)
        
        return outputs
    

class RMSNorm(torch.nn.Module):
    """
        1. Reference: https://github.com/meta-llama/llama/blob/1e8375848d3a3ebaccab83fd670b880864cf9409/llama/model.py#L34
    """
    def __init__(self, num_channels: int, eps: float = 1e-5):
        """
            Initialize the RMSNorm normalization layer.

            Args:
                dim (int): The dimension of the input tensor.
                eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

            Attributes:
                eps (float): A small value added to the denominator for numerical stability.
                weight (nn.Parameter): Learnable scaling parameter.
                
        """
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps

        self.weight = torch.nn.Parameter(torch.ones(self.num_channels))


    def _norm(self, x):
        """
            Apply the RMSNorm normalization to the input tensor.

            Args:
                x (torch.Tensor): The input tensor.

            Returns:
                torch.Tensor: The normalized tensor.
                
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


    def forward(self, x):
        """
            Forward pass through the RMSNorm layer.

            Args:
                x (torch.Tensor): The input tensor.

            Returns:
                torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    

    def __repr__(self):
        return f"{self.__class__.__name__}(num_channels={self.num_channels}, eps={self.eps})"


class EquivariantAdaNorm(torch.nn.Module):
    """等变特征上的 Adaptive LayerNorm（SCD v1）。

    包住现有的等变 norm，由条件向量额外产出 (shift, scale, gate)：
      - `scale` / `gate` 是不变标量，同一 (l, c) 内所有 m 分量共享同一个值
        （由 `expand_index` 保证），故乘上去仍等变；
      - `shift` 只加在 L=0，故加上去仍等变。

    `fc` 末层零初始化，且 `scale` / `gate` 以 `1 + delta` 读出 —— 未训练时本层
    逐位等价于原 norm，既有 ckpt 可无损续训（不同于 DiT 的 zero-init gate，
    那会让残差支在 step 0 完全关闭）。

    Args:
        scope: 'per_degree' 每个 degree 独立的 scale/gate；'shared' 全 degree 共享；
               'l0_only' 只调制 L=0（对齐参考实现 `dx*gate_x, dvec` 的严格形态）。
        use_node_feat: 是否把本节点的 L=0 特征拼进条件 MLP 的输入。参考实现如此
               （`Linear(2*dim, ...)` 硬编码了这个拼接），但论文正文/附录/Figure 3
               caption 均未描述这一点。
        detach_node_feat: 对上述 L=0 特征是否施加 `.detach()`。参考实现恒为 True。
               **保守力路径必须设 False**：能量通过调制系数依赖坐标，detach 会截断
               这条路径，使 autograd 的 `-dE/dpos` 不等于真实梯度（有限差分实测偏差
               比噪声地板高约 600 倍）。直接力预测不受影响。`use_node_feat=False`
               时本参数无意义。

    Returns:
        (x_out, gate)：`x_out` 形状与输入相同；`gate` 是**与 x 广播兼容**的张量，
        具体形状随 scope 而定 ——
          per_degree -> [N, (lmax+1)**2, C]
          shared     -> [N, 1, C]（不展开，靠广播，省一份完整激活）
          l0_only    -> [N, (lmax+1)**2, C]
        `cond is None` 时返回 (norm(x), None)。
    """

    _SCOPES = ('per_degree', 'shared', 'l0_only')

    def __init__(
        self,
        norm_type,
        lmax,
        num_channels,
        cond_channels,
        scope='per_degree',
        use_node_feat=True,
        detach_node_feat=True,
        eps=1e-5,
        affine=True,
        normalization='component'
    ):
        super().__init__()
        assert scope in self._SCOPES, f"unknown scope: {scope}"
        self.lmax = lmax
        self.num_channels = num_channels
        self.cond_channels = cond_channels
        self.scope = scope
        self.use_node_feat = use_node_feat
        self.detach_node_feat = detach_node_feat

        self.norm = get_normalization_layer(
            norm_type, lmax, num_channels, eps, affine, normalization
        )

        self.num_mod_degrees = (lmax + 1) if scope == 'per_degree' else 1
        out_channels = num_channels + 2 * self.num_mod_degrees * num_channels
        in_channels = cond_channels + (num_channels if use_node_feat else 0)

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_channels, num_channels),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(num_channels),
            torch.nn.Linear(num_channels, out_channels),
        )
        torch.nn.init.constant_(self.fc[-1].weight, 0.0)
        torch.nn.init.constant_(self.fc[-1].bias, 0.0)

        expand_index = torch.zeros([(lmax + 1) ** 2]).long()
        for l in range(lmax + 1):
            start_idx = l ** 2
            length = 2 * l + 1
            expand_index[start_idx : (start_idx + length)] = l
        self.register_buffer('expand_index', expand_index)

        l0_mask = torch.zeros(1, (lmax + 1) ** 2, 1)
        l0_mask[0, 0, 0] = 1.0
        self.register_buffer('l0_mask', l0_mask)

    def __repr__(self):
        return (f"{self.__class__.__name__}(lmax={self.lmax}, "
                f"num_channels={self.num_channels}, cond_channels={self.cond_channels}, "
                f"scope={self.scope}, use_node_feat={self.use_node_feat}, "
                f"detach_node_feat={self.detach_node_feat})")

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """兼容 AdaNorm 之前的 ckpt：把 <prefix>affine_* 重映射到 <prefix>norm.*。

        本类把原 norm 包了一层，键名多出一级 `norm.`。没有这个钩子，既有
        equiformer_v3 / DeNS 的 ckpt 在 strict=True 下会直接报错，strict=False
        下会静默把这些 affine 参数重置为初值——而 identity-init 的全部意义
        就是让既有 ckpt 能无损续训。
        """
        for key in ('affine_weight', 'affine_bias', 'balance_degree_weight'):
            old_key = prefix + key
            if old_key in state_dict:
                state_dict[prefix + 'norm.' + key] = state_dict.pop(old_key)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def _broadcast(self, v):
        """[N, num_mod_degrees, C] -> 可与 [N, (lmax+1)**2, C] 相乘的 delta。"""
        if self.scope == 'per_degree':
            return torch.index_select(v, dim=1, index=self.expand_index)
        if self.scope == 'shared':
            return v                                    # [N, 1, C]，契约允许不展开，靠广播省激活
        return v * self.l0_mask.to(v.dtype)             # l0_only：只有 L=0 非零

    def forward(self, x, cond=None):
        x = self.norm(x)
        if cond is None:
            return x, None

        if self.use_node_feat:
            node_feat = x.narrow(1, 0, 1).squeeze(1)
            if self.detach_node_feat:
                # 参考实现的默认行为。注意这会截断「能量经调制系数依赖坐标」这条
                # 路径，保守力下 autograd 的 -dE/dpos 不再是真实梯度。
                node_feat = node_feat.detach()
            inp = torch.cat([cond, node_feat], dim=-1)
        else:
            inp = cond

        out = self.fc(inp).to(x.dtype)
        c, d = self.num_channels, self.num_mod_degrees
        shift = out.narrow(1, 0, c)
        scale = out.narrow(1, c, d * c).view(-1, d, c)
        gate = out.narrow(1, c + d * c, d * c).view(-1, d, c)

        x = x * (1.0 + self._broadcast(scale))
        # shift 只进 L=0，out-of-place（避免 version-counter / make_fx 问题）
        x = x + shift.unsqueeze(1) * self.l0_mask.to(x.dtype)
        return x, 1.0 + self._broadcast(gate)