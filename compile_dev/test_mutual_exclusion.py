"""
test_mutual_exclusion.py
Gate test for Task 5: mutual exclusion between optim.use_compile and
model.enable_compile in EquiformerV3DeNSTrainer._setup_compile().

Three cases:
  1. use_compile=True,  enable_compile=False -> outer torch.compile IS invoked
  2. enable_compile=True, use_compile=False  -> outer NOT invoked, optimize_ddp=False
  3. both True                               -> raises ValueError

Run:
    bash scripts/compile_env.sh compile_dev/test_mutual_exclusion.py
"""

import sys
import torch
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Import the real trainer (the compile block lives in _setup_compile)
# ---------------------------------------------------------------------------
from experimental.trainers.equiformer_v3_dens_trainer import EquiformerV3DeNSTrainer


def _make_stub(use_compile: bool, enable_compile: bool) -> EquiformerV3DeNSTrainer:
    """
    Create a minimal EquiformerV3DeNSTrainer instance that skips __init__
    entirely and sets only the two attributes needed by _setup_compile:
      - self.config
      - self.model
    """
    trainer = object.__new__(EquiformerV3DeNSTrainer)
    trainer.config = {
        'optim': {'use_compile': use_compile},
        'model': {'enable_compile': enable_compile},
    }
    trainer.model = MagicMock(name='model')
    return trainer


# ---------------------------------------------------------------------------
# Case 1: use_compile=True, enable_compile=False -> outer compile invoked
# ---------------------------------------------------------------------------
def test_case1_outer_compile_invoked():
    dummy_compiled = MagicMock(name='compiled_model')
    with patch('torch.compile', return_value=dummy_compiled) as mock_compile:
        trainer = _make_stub(use_compile=True, enable_compile=False)
        trainer._setup_compile()
        assert mock_compile.called, (
            "FAIL case1: torch.compile was NOT called but should be when "
            "use_compile=True and enable_compile=False"
        )
        # model should be replaced with the compiled version
        assert trainer.model is dummy_compiled, (
            "FAIL case1: trainer.model was not replaced by torch.compile result"
        )
    print("PASS case1: outer torch.compile invoked (use_compile=True, enable_compile=False)")


# ---------------------------------------------------------------------------
# Case 2: enable_compile=True, use_compile=False -> outer NOT invoked, ddp=False
# ---------------------------------------------------------------------------
def test_case2_inner_compile_only():
    # Reset optimize_ddp to True so we can detect the change
    torch._dynamo.config.optimize_ddp = True

    with patch('torch.compile', return_value=MagicMock()) as mock_compile:
        trainer = _make_stub(use_compile=False, enable_compile=True)
        trainer._setup_compile()
        assert not mock_compile.called, (
            "FAIL case2: torch.compile WAS called but should NOT be when "
            "enable_compile=True and use_compile=False"
        )

    assert torch._dynamo.config.optimize_ddp is False, (
        "FAIL case2: optimize_ddp was not set to False when enable_compile=True"
    )
    print("PASS case2: outer torch.compile NOT invoked, optimize_ddp=False (enable_compile=True, use_compile=False)")


# ---------------------------------------------------------------------------
# Case 3: both True -> raises ValueError
# ---------------------------------------------------------------------------
def test_case3_both_raises():
    trainer = _make_stub(use_compile=True, enable_compile=True)
    raised = False
    try:
        trainer._setup_compile()
    except ValueError as exc:
        raised = True
        assert 'use_compile' in str(exc) or '互斥' in str(exc), (
            f"FAIL case3: ValueError raised but message unexpected: {exc}"
        )
    assert raised, (
        "FAIL case3: expected ValueError when both use_compile=True and enable_compile=True, but none was raised"
    )
    print("PASS case3: ValueError raised when both True (use_compile=True, enable_compile=True)")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def main():
    cases = [
        test_case1_outer_compile_invoked,
        test_case2_inner_compile_only,
        test_case3_both_raises,
    ]
    failed = False
    for fn in cases:
        try:
            fn()
        except Exception as exc:
            print(f"FAIL {fn.__name__}: {exc}")
            failed = True

    if failed:
        sys.exit(1)
    print("\nAll 3 mutual-exclusion cases PASSED.")


if __name__ == '__main__':
    main()
