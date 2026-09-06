"""Test suite for PointCloudEvaluator metrics (CD, EMD, MMD, TMD, JSD).

Mirrors utils/sensat_metrics.py.
"""
import os
import sys
import torch

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.sensat_metrics import PointCloudEvaluator


def test_sensat_metrics():
    print("\n--- PointCloudEvaluator Sanity Test (using random tensors) ---\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on: {device}\n")

    # Simulate k=10 generated completions and M=50 reference shapes
    # Shape [B, N, 3] with coordinates in [-1, 1] (unit sphere — like SensatUrban output)
    torch.manual_seed(42)
    generated = torch.rand(10, 1024, 3) * 2 - 1   # [10, 1024, 3], range [-1, 1]
    reference = torch.rand(50, 1024, 3) * 2 - 1   # [50, 1024, 3], range [-1, 1]

    evaluator = PointCloudEvaluator(device=device, k=10, verbose=True)
    results   = evaluator.evaluate(generated, reference)

    print("Raw results dictionary:")
    print(results)

    assert "CD" in results and results["CD"] > 0
    assert "EMD" in results and results["EMD"] > 0
    assert "MMD" in results and results["MMD"] > 0
    assert "TMD" in results and results["TMD"] > 0
    assert "JSD" in results and results["JSD"] >= 0

    print("\n--- Sanity test complete ---")


if __name__ == "__main__":
    test_sensat_metrics()
