import torch

from fairchem.core.common.registry import registry
from fairchem.core.common.utils import conditional_grad

import e3nn
import math

from .so3 import SO3Linear
from .layer_norm import (
    RMSNorm,
    get_normalization_layer
)
from .transformer_block import EquivariantGraphAttention, FeedForwardNetwork
from .equiformer_v3 import EquiformerV3_OC


# Statistics of IS2RE 100K
_AVG_NUM_NODES = 77.81317
_AVG_DEGREE = 23.395238876342773    # IS2RE: 100k, max_radius = 5, max_neighbors = 100


@registry.register_model("equiformer_v3_dens")
class EquiformerV3DeNS_OC(EquiformerV3_OC):
    """
    Args:
        use_pbc (bool):         Use periodic boundary conditions
        use_pbc_single (bool, optional):    Process batch PBC graphs one at a time.
                                            This argument is set to True when training on MPTraj.
        otf_graph (bool):       Compute graph On The Fly (OTF)

        regress_forces (bool):  Compute forces
        regress_stress (bool):  Compute stress
        direct_prediction (bool):   Whether to use direct methods to predict forces and stress

        max_neighbors (int):    Maximum number of neighbors per atom
        max_radius (float):     Maximum distance between nieghboring atoms in Angstroms
        num_radial_basis (int): Number of radial basis functions
        max_num_elements (int): Maximum atomic number

        num_layers (int):           Number of layers in the GNN
        num_channels (int):         Number of channels in node embeddings
        attn_hidden_channels (int): Number of hidden channels in equivariant graph attention
        num_heads (int):            Number of attention heads
        attn_alpha_channels (int):  Number of channels for alpha vector in each attention head
        attn_value_channels (int):  Number of channels for value vector in each attention head
        ffn_hidden_channels (int):  Number of hidden channels in feedforward network
        norm_type (str):            Type of normalization layer 
                                    (['sep_layer_norm', 'sep_layer_norm_liger', 'merge_layer_norm', 
                                    'merge_layer_norm_attn_rms_norm', 'merge_rms_norm'])

        lmax (int):                 Maximum degrees (l)
        mmax (int):                 Maximum order (m)
        attn_grid_resolution_list (list:int):      
                                    Grid resolution list in class `SO3Grid` in attention
        ffn_grid_resolution_list (list:int):      
                                    Grid resolution list in class `SO3Grid` in feedforward network
        
        edge_channels (int):                Number of channels for edge-wise invariant features
        use_atom_edge_embedding (bool):     Whether to use atomic embedding along with relative distance for edge scalar features
        use_envelope (bool):        Whether to apply an envelope function to attention
        
        attn_activation (str):      Type of activation function in equivariant graph attention
        use_attn_renorm (bool):     Whether to re-normalize attention weights
        use_add_merge (bool):       Default: False
                                    If True, use addition to merge the source/target node features instead of concat, 
                                    which can save 2x compute when rotating with Wigner-D matrices.
        use_rad_l_parametrization (bool):
                                    Default: True
                                    If True, all the m components within the same type-L vector will share the same
                                    weight from the radial function.
        softcap (float):            Default: None
                                    If not None, use soft capping to limit the range of attention logits to
                                    [- `softcap`, + `softcap`].
        attn_eps (float):           Default: 1e-16
                                    Epsilon value used in the softmax operation of attention
        ffn_activation (str):       Type of activation function for feedforward network
        use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs for FFNs.

        use_gate_force_head (bool): If `True`, use `GateActivation` in the equivariant attention of the force prediction head.
        
        alpha_drop (float):         Dropout rate for the hidden features in non-linear MLP attention
        attn_mask_rate (float):     Mask rate for neighbors considered in attention
        attn_weights_drop (float):  Dropout rate for attention weights
        value_drop (float):         Dropout rate for the hidden features in non-linear value vectors
        drop_path_rate (float):     Drop path rate
        proj_drop (float):          Dropout rate for outputs of attention and FFN in Transformer blocks
        ffn_drop (float):           Dropout rate for the hidden features in FFN
        use_head_reg (bool):        Whether to apply regularization to output head (dummy argument for backend compatibility)
                
        gradient_checkpointing_block_list (list):
                                    A list indicating which block we apply gradient/activation checkpointing to save memory.

        avg_num_nodes (float):   Normalization factor for sum aggregation over nodes
        avg_degree (float):      Normalization factor for sum aggregation over edges

        enforce_max_neighbors_strictly (bool):      When edges are subselected based on the `max_neighbors` arg, arbitrarily select amongst equidistant / degenerate edges to have exactly the correct number.
    """
    def __init__(
        self,
        
        use_pbc=True,
        use_pbc_single=False, 
        otf_graph=True,

        regress_forces=True,
        regress_stress=False,
        direct_prediction=True,

        max_neighbors=20,
        max_radius=12.0,
        num_radial_basis=600,
        max_num_elements=128,

        num_layers=12,
        num_channels=128,
        attn_hidden_channels=64,
        num_heads=8,
        attn_alpha_channels=32,
        attn_value_channels=16,
        ffn_hidden_channels=128,
        norm_type='merge_layer_norm',

        lmax=6,
        mmax=2,
        attn_grid_resolution_list=[20, 8],
        ffn_grid_resolution_list=[20, 20],

        edge_channels=128,
        use_atom_edge_embedding=True,
        use_envelope=True,

        attn_activation='sep-merge_gates2_swiglu',
        use_attn_renorm=True,
        use_add_merge=False,
        use_rad_l_parametrization=True,
        softcap=None,
        attn_eps=1e-16,
        ffn_activation='sep-merge_gates2_swiglu',
        use_grid_mlp=True,

        use_gate_force_head=True,

        alpha_drop=0.0,
        attn_mask_rate=0.0,
        attn_weights_drop=0.1,
        value_drop=0.0,
        drop_path_rate=0.05,
        proj_drop=0.0,
        ffn_drop=0.0,
        use_head_reg=False,
        
        gradient_checkpointing_block_list=None,

        avg_num_nodes=_AVG_NUM_NODES,
        avg_degree=_AVG_DEGREE,

        enforce_max_neighbors_strictly=True,

        enable_compile: bool = False,
        compile_dynamic: bool = False,
        use_amp: bool = False,

        # ---- DPA4 ablation switches (forwarded to the parent; defaults = pre-change) ---------
        envelope_type: str = 'equiformerv3_c2',
        envelope_exponent: int = 5,
        attn_softmax_type: str = 'equiformerv3',
        focus_compete_groups: int = 0,
    ):
        super().__init__(
            use_pbc,
            use_pbc_single,
            otf_graph,

            regress_forces,
            regress_stress,
            direct_prediction,

            max_neighbors,
            max_radius,
            num_radial_basis,
            max_num_elements,

            num_layers,
            num_channels,
            attn_hidden_channels,
            num_heads,
            attn_alpha_channels,
            attn_value_channels,
            ffn_hidden_channels,
            norm_type,

            lmax,
            mmax,
            attn_grid_resolution_list,
            ffn_grid_resolution_list,

            edge_channels,
            use_atom_edge_embedding,
            use_envelope,

            attn_activation,
            use_attn_renorm,
            use_add_merge,
            use_rad_l_parametrization,
            softcap,
            attn_eps,
            ffn_activation,
            use_grid_mlp,

            use_gate_force_head,

            alpha_drop,
            attn_mask_rate,
            attn_weights_drop,
            value_drop,
            drop_path_rate,
            proj_drop,
            ffn_drop,
            use_head_reg,

            gradient_checkpointing_block_list,

            avg_num_nodes,
            avg_degree,

            enforce_max_neighbors_strictly,

            enable_compile,
            compile_dynamic,
            use_amp,
            # Passed BY KEYWORD -- everything above is positional, so any future insertion in
            # the parent signature must also go at its end (see the note there).
            envelope_type=envelope_type,
            envelope_exponent=envelope_exponent,
            attn_softmax_type=attn_softmax_type,
            focus_compete_groups=focus_compete_groups,
        )

        # Force encoding
        self.irreps_sh = e3nn.o3.Irreps.spherical_harmonics(lmax=self.lmax, p=1)
        self.force_embedding = SO3Linear(
            in_features=1,
            out_features=self.num_channels,
            lmax=self.lmax,
            bias=True
        )
        
        if self.regress_forces:
            self.dens_block = EquivariantGraphAttention(
                num_in_channels=self.num_channels,
                num_hidden_channels=self.attn_hidden_channels,
                num_heads=self.num_heads,
                attn_alpha_channels=self.attn_alpha_channels,
                attn_value_channels=self.attn_value_channels,
                num_out_channels=1,
                lmax=self.lmax,
                mmax=self.mmax,
                so3_rotation=self.so3_rotation,
                grid_resolution_list=self.attn_grid_resolution_list,
                max_num_elements=self.max_num_elements,
                edge_channels_list=self.edge_channels_list,
                use_atom_edge_embedding=self.use_atom_edge_embedding,
                activation=('sep_s2' if not self.use_gate_force_head else 'gate'),
                use_attn_renorm=self.use_attn_renorm,
                use_add_merge=self.use_add_merge,
                use_rad_l_parametrization=self.use_rad_l_parametrization,
                softcap=self.softcap,
                eps=self.attn_eps,
                alpha_drop=0.0, 
                attn_mask_rate=0.0, 
                attn_weights_drop=0.0,
                value_drop=0.0
            )
            if 'rms_norm' in norm_type:
                if self.dens_block.alpha_norm is not None:
                    del self.dens_block.alpha_norm
                    self.dens_block.alpha_norm = RMSNorm(self.attn_alpha_channels)

        self.apply(self._init_weights)


    def core_compute(
        self,
        atomic_numbers,
        edge_distance,
        edge_distance_vec,
        edge_index,
        batch,
        force_embedding,
    ):
        """Pure tensor-in / tensor-out compute body (DeNS variant).

        Like the base core_compute (takes RAW edge_distance + edge_distance_vec
        and runs _forward_edge internally so pos->Wigner is captured when
        traced/compiled) but adds force_embedding to the node embedding before
        the transformer blocks. The caller provides force_embedding (from
        _forward_dens_force_encoding). Energy aggregation and force/stress/dens
        heads remain in the callers.
        """
        edge_distance, edge_envelope_weight = self._forward_edge(edge_distance, edge_distance_vec)
        source_atomic_numbers = atomic_numbers[edge_index[0]]
        target_atomic_numbers = atomic_numbers[edge_index[1]]
        x = self._forward_embedding(atomic_numbers, edge_distance, edge_index, edge_envelope_weight)
        x = x + force_embedding
        x_scalar, x = self._forward_blocks(
            x,
            source_atomic_numbers,
            target_atomic_numbers,
            edge_distance,
            edge_index,
            edge_envelope_weight,
            batch,
        )
        return x_scalar, x, edge_distance, edge_envelope_weight


    def _forward_direct(self, data):
        self.batch_size = len(data.natoms)
        self.dtype = data.pos.dtype
        self.device = data.pos.device

        (
            edge_index,
            edge_distance,
            edge_distance_vec,
            cell_offsets,
            _,  # cell offset distances
            neighbors,
        ) = self.generate_graph(
            data,
            enforce_max_neighbors_strictly=self.enforce_max_neighbors_strictly,
            use_pbc_single=self.use_pbc_single
        )

        atomic_numbers = data.atomic_numbers.long()
        source_atomic_numbers = atomic_numbers[edge_index[0]]
        target_atomic_numbers = atomic_numbers[edge_index[1]]

        force_embedding, noise_mask_tensor, dens_batch_mask_tensor, dens_mask_tensor = self._forward_dens_force_encoding(data)
        compute = self.core_compute
        if self.enable_compile:
            if self._compiled_core is None:
                from fairchem.core.common.compile_utils import plain_compile
                self._compiled_core = plain_compile(self.core_compute, dynamic=self.compile_dynamic)
            compute = self._compiled_core
        x_scalar, x, edge_distance, edge_envelope_weight = compute(
            atomic_numbers,
            edge_distance,
            edge_distance_vec,
            edge_index,
            data.batch,
            force_embedding,
        )

        outputs = {}

        # Energy prediction
        node_energy = self.energy_block(x_scalar)
        energy = torch.zeros(self.batch_size, device=node_energy.device, dtype=node_energy.dtype)
        energy.index_add_(0, data.batch, node_energy.view(-1))
        energy = energy / self.avg_num_nodes
        outputs['energy'] = energy

        # Force prediction
        if self.regress_forces:
            forces = self.force_block(
                x,
                source_atomic_numbers,
                target_atomic_numbers,
                edge_distance,
                edge_index,
                edge_envelope_weight
            )
            forces = forces.narrow(1, 1, 3)
            forces = forces.view(-1, 3)

            # for DeNS
            denoising_pos_vec = self.dens_block(
                x,
                source_atomic_numbers,
                target_atomic_numbers,
                edge_distance,
                edge_index,
                edge_envelope_weight
            )
            denoising_pos_vec = denoising_pos_vec.narrow(1, 1, 3)
            denoising_pos_vec = denoising_pos_vec.view(-1, 3)

            outputs['forces'] = forces * (~noise_mask_tensor) + denoising_pos_vec * noise_mask_tensor
        
        # Stress Prediction
        if self.regress_stress:
            stress = self.stress_block(
                x,
                batch_size=self.batch_size,
                batch=data.batch
            )
            outputs['stress'] = stress * (~dens_batch_mask_tensor) # not predict stress during DeNS

        return outputs
    

    @conditional_grad(torch.enable_grad())
    def _forward_gradient(self, data):
        """
            1.  We have additional `@conditional_grad` as the decorator since the decorator might not be compatible with
                `torch.compile()` in direct methods.
        """
        # 仅训练态走保守力编译区（该区以 create_graph=True 编译整段双反向，供外层 param 反向用）。
        # eval 无外层 backward，改走下方 eager 路径（create_graph=self.training=False）—— 不建双反向图、
        # 不触发编译，消除首次 eval 的显存尖峰。对齐 DPA4 should_use_compile 的 self.training 门控。
        if self.enable_compile and self.training:
            return self._conservative_compiled_forward(data)

        self.batch_size = len(data.natoms)
        self.dtype = data.pos.dtype
        self.device = data.pos.device

        """
            For gradient methods
        """
        displacement = None
        orig_cell = None
        if self.regress_stress and self.regress_forces:
            displacement = torch.zeros(
                (3, 3),
                dtype=self.dtype,
                device=self.device,
            )
            displacement = displacement.view(-1, 3, 3).expand(self.batch_size, 3, 3)
            displacement.requires_grad = True
            symmetric_displacement = 0.5 * (
                displacement + displacement.transpose(-1, -2)
            )

            data.pos.requires_grad = True
            data.pos = data.pos + torch.bmm(
                data.pos.unsqueeze(-2),
                torch.index_select(symmetric_displacement, 0, data.batch)
            ).squeeze(-2)

            orig_cell = data.cell
            data.cell = data.cell + torch.bmm(
                data.cell, symmetric_displacement
            )
        elif not self.regress_stress and self.regress_forces:
            data.pos.requires_grad = True

        (
            edge_index,
            edge_distance,
            edge_distance_vec,
            cell_offsets,
            _,  # cell offset distances
            neighbors,
        ) = self.generate_graph(
            data,
            enforce_max_neighbors_strictly=self.enforce_max_neighbors_strictly,
            use_pbc_single=self.use_pbc_single
        )

        atomic_numbers = data.atomic_numbers.long()
        source_atomic_numbers = atomic_numbers[edge_index[0]]
        target_atomic_numbers = atomic_numbers[edge_index[1]]

        force_embedding, noise_mask_tensor, dens_batch_mask_tensor, dens_mask_tensor = self._forward_dens_force_encoding(data)
        x_scalar, x, edge_distance, edge_envelope_weight = self.core_compute(
            atomic_numbers,
            edge_distance,
            edge_distance_vec,
            edge_index,
            data.batch,
            force_embedding,
        )

        outputs = {}

        # Energy prediction
        node_energy = self.energy_block(x_scalar)
        energy = torch.zeros(self.batch_size, device=node_energy.device, dtype=node_energy.dtype)
        energy.index_add_(0, data.batch, node_energy.view(-1))
        energy = energy / self.avg_num_nodes
        outputs['energy'] = energy

        if self.regress_stress and self.regress_forces:
            # Stress and forces prediction
            grads = torch.autograd.grad(
                [energy.sum()],
                [data.pos, displacement],
                create_graph=self.training,
            )
            forces = torch.neg(grads[0])
            virial = grads[1].view(-1, 3, 3)
            volume = torch.det(data.cell).abs().unsqueeze(-1)
            stress = virial / volume.view(-1, 1, 1)
            virial = torch.neg(virial)
            outputs['forces'] = forces
            outputs['stress'] = stress.view(-1, 9) * (~dens_batch_mask_tensor) # not predict stress during DeNS
            data.cell = orig_cell
        elif not self.regress_stress and self.regress_forces:
            forces = (
                -1
                * torch.autograd.grad(
                    energy.sum(), data.pos, create_graph=self.training
                )[0]
            )
            outputs['forces'] = forces
        
        # for DeNS
        if self.regress_forces:
            denoising_pos_vec = self.dens_block(
                x,
                source_atomic_numbers,
                target_atomic_numbers,
                edge_distance,
                edge_index,
                edge_envelope_weight
            )
            denoising_pos_vec = denoising_pos_vec.narrow(1, 1, 3)
            denoising_pos_vec = denoising_pos_vec.view(-1, 3)
            outputs['forces'] = forces * (~noise_mask_tensor) + denoising_pos_vec * noise_mask_tensor # not predict forces during DeNS

        return outputs


    def _conservative_compiled_forward(self, data):
        """make_fx-compiled conservative force/stress (Task 4, DeNS variant).

        DeNS twin of EquiformerV3_OC._conservative_compiled_forward. The whole
        pos -> edv -> Wigner -> energy -> autograd.grad double-backward runs
        inside the traced region (CompiledForceRegion: make_fx -> strip_detach
        -> rebuild -> inductor, per-shape cached). The DeNS denoising head
        (dens_block) stays EAGER — when enable_compile we run one extra eager
        core_compute to get the equivariant feature ``x`` it needs.

        Stale-bake guard: every per-batch tensor (atomic_numbers, edge_index,
        cell_offsets, cell, batch, force_embedding) is an explicit core_fn input,
        NOT a closure capture. force_embedding is computed eagerly (it is constant
        wrt pos) and passed in as ``fe``; the traced region stays differentiable
        wrt fe, so the SO3Linear force_embedding's params still receive gradients
        through the outer backward. Only live module refs (self.core_compute,
        self.energy_block) stay in the closure.
        """
        from fairchem.core.common.compile_utils import (
            CompiledForceRegion,
            make_prime_graph_example,
        )

        self.batch_size = len(data.natoms)
        self.dtype = data.pos.dtype
        self.device = data.pos.device

        (
            edge_index,
            edge_distance,
            edge_distance_vec,
            cell_offsets,
            _,
            neighbors,
        ) = self.generate_graph(
            data,
            enforce_max_neighbors_strictly=self.enforce_max_neighbors_strictly,
            use_pbc_single=self.use_pbc_single,
        )

        atomic_numbers = data.atomic_numbers.long()
        source_atomic_numbers = atomic_numbers[edge_index[0]]
        target_atomic_numbers = atomic_numbers[edge_index[1]]

        force_embedding, noise_mask_tensor, dens_batch_mask_tensor, dens_mask_tensor = \
            self._forward_dens_force_encoding(data)

        energy_block = self.energy_block
        avg_num_nodes = self.avg_num_nodes

        def _energy(pos_p, cell_p, an, ei, co, batch, fe, n_sys):
            co = co.reshape(ei.shape[1], -1)
            src, dst = ei[0], ei[1]
            cell_e = cell_p.index_select(0, batch.index_select(0, src))
            shifts = torch.einsum("ej,ejk->ek", co, cell_e)
            edv = pos_p.index_select(0, src) - pos_p.index_select(0, dst) + shifts
            ed = torch.linalg.norm(edv, dim=-1)
            x_scalar, _x, _, _ = self.core_compute(an, ed, edv, ei, batch, fe)
            node_e = energy_block(x_scalar).view(-1)
            energy = torch.zeros(n_sys, device=node_e.device, dtype=node_e.dtype)
            energy.index_add_(0, batch, node_e)
            energy = energy / avg_num_nodes
            return energy

        def core_fn_stress(pos, disp, an, ei, co, cell, batch, fe):
            sym = 0.5 * (disp + disp.transpose(-1, -2))
            pos_p = pos + torch.bmm(
                pos.unsqueeze(-2), torch.index_select(sym, 0, batch)
            ).squeeze(-2)
            cell_p = cell + torch.bmm(cell, sym)
            energy = _energy(pos_p, cell_p, an, ei, co, batch, fe, cell.shape[0])
            grads = torch.autograd.grad([energy.sum()], [pos, disp], create_graph=True)
            forces = torch.neg(grads[0])
            virial = grads[1].view(-1, 3, 3)
            volume = torch.det(cell).abs().unsqueeze(-1)
            stress = (virial / volume.view(-1, 1, 1)).view(-1, 9)
            return energy, forces, stress

        def core_fn_force(pos, an, ei, co, cell, batch, fe):
            energy = _energy(pos, cell, an, ei, co, batch, fe, cell.shape[0])
            forces = torch.neg(
                torch.autograd.grad(energy.sum(), pos, create_graph=True)[0]
            )
            return energy, forces

        if self._compiled_region is None:
            self._compiled_region = CompiledForceRegion(
                dynamic=self.compile_dynamic, optimize_ddp=False
            )

        dtype = data.pos.dtype
        an = atomic_numbers
        ei = edge_index
        co = cell_offsets.to(dtype)
        cell = data.cell.to(dtype)
        batch = data.batch
        fe = force_embedding
        pos = data.pos.detach().requires_grad_(True)

        dyn = self.compile_dynamic

        def _prime(stress):
            # dynamic=True only: append a prime force_embedding leaf so the trace
            # example matches the real (pos[,disp],an,ei,co,cell,batch,fe) layout.
            pe = make_prime_graph_example(pos.device, dtype, stress=stress)
            fe_p = torch.zeros(pe[0].shape[0], *fe.shape[1:], device=pos.device, dtype=dtype)
            return (*pe, fe_p)

        outputs = {}
        if self.regress_stress and self.regress_forces:
            disp = torch.zeros(
                (cell.shape[0], 3, 3), device=pos.device, dtype=dtype
            ).requires_grad_(True)
            energy, forces, stress = self._compiled_region(
                core_fn_stress,
                (pos, disp, an, ei, co, cell, batch, fe),
                trace_example=_prime(True) if dyn else None,
                dynamic_dims=[(pos, [0]), (disp, [0]), (an, [0]), (ei, [1]),
                              (co, [0]), (cell, [0]), (batch, [0]), (fe, [0])] if dyn else None,
            )
            outputs['energy'] = energy
            outputs['forces'] = forces
            # stress not predicted during DeNS (per-system mask), eager outside region
            outputs['stress'] = stress * (~dens_batch_mask_tensor)
        elif self.regress_forces:
            energy, forces = self._compiled_region(
                core_fn_force,
                (pos, an, ei, co, cell, batch, fe),
                trace_example=_prime(False) if dyn else None,
                dynamic_dims=[(pos, [0]), (an, [0]), (ei, [1]), (co, [0]),
                              (cell, [0]), (batch, [0]), (fe, [0])] if dyn else None,
            )
            outputs['energy'] = energy
            outputs['forces'] = forces

        # DeNS denoising stays EAGER: one extra eager core_compute gives the
        # equivariant feature x for dens_block. Mirrors the eager _forward_gradient
        # force/denoising combination exactly (same noise mask), so dens_block /
        # backbone gradients via this path are identical eager-vs-compiled.
        if self.regress_forces:
            _x_scalar, x, edge_distance_exp, edge_envelope_weight = self.core_compute(
                atomic_numbers,
                edge_distance,
                edge_distance_vec,
                edge_index,
                data.batch,
                force_embedding,
            )
            denoising_pos_vec = self.dens_block(
                x,
                source_atomic_numbers,
                target_atomic_numbers,
                edge_distance_exp,
                edge_index,
                edge_envelope_weight,
            )
            denoising_pos_vec = denoising_pos_vec.narrow(1, 1, 3).view(-1, 3)
            outputs['forces'] = (
                outputs['forces'] * (~noise_mask_tensor)
                + denoising_pos_vec * noise_mask_tensor
            )

        return outputs


    @torch._dynamo.disable
    def _generate_dens_data(self, data):
        num_atoms = len(data.atomic_numbers)
        if hasattr(data, "denoising_pos_forward") and data.denoising_pos_forward:
            force_data = data.forces
            if hasattr(data, "noise_mask"):
                noise_mask_tensor = data.noise_mask.view(-1, 1)
            else:
                noise_mask_tensor = torch.ones((num_atoms, 1), dtype=torch.bool, device=self.device)
            if hasattr(data, "dens_batch_mask"):
                dens_batch_mask_tensor = data.dens_batch_mask.view(-1, 1)
            else:
                dens_batch_mask_tensor = torch.ones((len(data.natoms), 1), dtype=torch.bool, device=self.device)
            dens_mask_tensor = torch.ones((1, 1), dtype=torch.bool, device=self.device)
        else:
            force_data = torch.zeros((num_atoms, 3), dtype=self.dtype, device=self.device)
            noise_mask_tensor = torch.zeros((num_atoms, 1), dtype=torch.bool, device=self.device)
            dens_batch_mask_tensor = torch.zeros((len(data.natoms), 1), dtype=torch.bool, device=self.device)
            dens_mask_tensor = torch.zeros((1, 1), dtype=torch.bool, device=self.device)
        force_sh = e3nn.o3.spherical_harmonics(
            l=self.irreps_sh,
            x=force_data,
            normalize=True,
            normalization='component'
        )
        
        return force_data, force_sh, noise_mask_tensor, dens_batch_mask_tensor, dens_mask_tensor


    def _forward_dens_force_encoding(self, data):
        force_data, force_sh, noise_mask_tensor, dens_batch_mask_tensor, dens_mask_tensor = self._generate_dens_data(data)
        force_norm = force_data.norm(dim=-1, keepdim=True)
        force_norm = force_norm / math.sqrt(3.0)
        force_embedding = force_sh * force_norm
        force_embedding = force_embedding.view(force_embedding.shape[0], -1, 1)
        force_embedding = self.force_embedding(force_embedding)
        noise_mask_tensor = noise_mask_tensor.view(-1, 1, 1)
        force_embedding = force_embedding * noise_mask_tensor
        noise_mask_tensor = noise_mask_tensor.view(-1, 1)

        return force_embedding, noise_mask_tensor, dens_batch_mask_tensor, dens_mask_tensor