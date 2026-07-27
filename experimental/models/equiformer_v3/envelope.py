import torch


class PolynomialEnvelope(torch.nn.Module):
    """
        1.  Polynomial envelope function that ensures a smooth cutoff.
        2.  Reference: https://github.com/facebookresearch/fairchem/blob/518d0ea12110548bd5ffaf9a43060b8eae152e13/src/fairchem/core/models/esen/nn/radial.py#L22
    """
    def __init__(self, cutoff: float = 6.0, exponent: int = 5) -> None:
        super().__init__()
        assert exponent > 0
        self.cutoff = float(cutoff)
        self.exponent = exponent
        self.p: float = float(exponent)
        self.a: float = -(self.p + 1) * (self.p + 2) / 2
        self.b: float = self.p * (self.p + 2)
        self.c: float = -self.p * (self.p + 1) / 2


    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        d_scaled = distance / self.cutoff
        env_val = (
            1
            + self.a * d_scaled**self.p
            + self.b * d_scaled ** (self.p + 1)
            + self.c * d_scaled ** (self.p + 2)
        )
        outputs = torch.where(d_scaled < 1, env_val, torch.zeros_like(d_scaled))
        outputs = outputs.view(-1, 1)
        return outputs
    

    def extra_repr(self):
        return 'cutoff={}, exponent={}'.format(self.cutoff, self.exponent)


class C3CutoffEnvelope(torch.nn.Module):
    """
        C3-continuous polynomial cutoff envelope (DPA4 / SeZM).

        Ported (math kept verbatim) from the read-only reference
            ../DPA4/deepmd-kit/deepmd/pt/model/descriptor/sezm_nn/radial.py `C3CutoffEnvelope`
            commit 99c1ece2e5087c77267fba4ca84932b53621e42c
        via mlip-forge `_dpa4_ops/envelope.py`. Only the deepmd dependency (`env.DEVICE`) was
        stripped and the output reshaped to `[E, 1]` to match `PolynomialEnvelope`'s contract,
        so this is a drop-in replacement.

        WHY (the entire reason this class exists) -- with x = r / rcut:

            PolynomialEnvelope (above, DimeNet 3-term, p=5)
                u(1) = u'(1) = u''(1) = 0,  u'''(1) = -210 / rcut^3 != 0     -> only C2
            C3CutoffEnvelope (4-term, p=5)
                E(1) = E'(1) = E''(1) = E'''(1) = 0, E''''(1) != 0           -> C3

        kappa_SRME depends on THIRD-order force constants (phonon-phonon anharmonic
        scattering). A C2 envelope means the PES third derivative JUMPS at the cutoff
        radius; C3 pushes that jump out to fourth order. Verified independently here with
        autograd (not taken from any docstring): the -210 and the C3 zero both reproduce.

        Coefficients are uniquely determined by "E(1)=0 and the first three derivatives
        vanish":
            a = -(p+1)(p+2)(p+3)/6   b =  p(p+2)(p+3)/2
            c = -p(p+1)(p+3)/2       d =  p(p+1)(p+2)/6
        p=5 -> E(x) = 1 - 56x^5 + 140x^6 - 120x^7 + 35x^8.
    """
    def __init__(self, cutoff: float = 6.0, exponent: int = 5) -> None:
        super().__init__()
        assert cutoff > 0.0
        assert exponent > 0
        self.cutoff = float(cutoff)
        self.exponent = int(exponent)
        self.p: float = float(self.exponent)
        self.a: float = -((self.p + 1) * (self.p + 2) * (self.p + 3)) / 6.0
        self.b: float = (self.p * (self.p + 2) * (self.p + 3)) / 2.0
        self.c: float = -(self.p * (self.p + 1) * (self.p + 3)) / 2.0
        self.d: float = (self.p * (self.p + 1) * (self.p + 2)) / 6.0


    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        # DPA4 clamps to [0, 1] first, then evaluates with Horner, then applies a hard mask.
        # clamp(max=1) already makes E=0 for x>=1 (since E(1)=0); the mask is redundant but
        # harmless -- E and its first three derivatives all vanish at x=1, so it introduces
        # no discontinuity.
        d_scaled = (distance / self.cutoff).clamp(min=0.0, max=1.0)
        poly = self.a + d_scaled * (self.b + d_scaled * (self.c + d_scaled * self.d))
        env_val = 1 + d_scaled.pow(self.p) * poly
        outputs = env_val * (d_scaled < 1.0).to(distance.dtype)
        outputs = outputs.view(-1, 1)
        return outputs


    def extra_repr(self):
        return 'cutoff={}, exponent={}, continuity=C3'.format(self.cutoff, self.exponent)


#: Registry for `envelope_type`. Keys are the strings accepted in the yml `model:` block.
ENVELOPE_REGISTRY = {
    'equiformerv3_c2': PolynomialEnvelope,   # DEFAULT -- current/native behaviour, bit-identical
    'dpa4_c3':         C3CutoffEnvelope,     # DPA4 ablation arm (D4)
}


def build_envelope(envelope_type: str = 'equiformerv3_c2', cutoff: float = 6.0, exponent: int = 5):
    """
        Single factory for the cutoff envelope.

        There is exactly ONE envelope construction site in the training path
        (`equiformer_v3.py`), whose result is threaded to every consumer as
        `edge_envelope_weight`. Routing that one site through this factory keeps it that way:
        it is impossible for two different cutoffs to coexist inside one forward pass.
    """
    if envelope_type not in ENVELOPE_REGISTRY:
        raise ValueError(
            "unknown envelope_type={!r}; expected one of {}".format(
                envelope_type, sorted(ENVELOPE_REGISTRY)
            )
        )
    return ENVELOPE_REGISTRY[envelope_type](cutoff=cutoff, exponent=exponent)