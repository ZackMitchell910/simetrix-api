# src/quantum_engine.py
from __future__ import annotations
import os, math
from typing import Sequence, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as IBMSampler

# Optional: IBM Runtime (only used if properly configured)
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as IBMSampler
    _IBM_AVAILABLE = True
except Exception:
    QiskitRuntimeService = None
    IBMSampler = None
    _IBM_AVAILABLE = False
def _ibm_sampler():
    token = os.getenv("QISKIT_IBM_TOKEN", "").strip()
    if token:
        svc = QiskitRuntimeService(channel="ibm_quantum", token=token)
    else:
        svc = QiskitRuntimeService()  # uses ~/.qiskit on dev
    return IBMSampler(session=None, service=svc)
def _nearest_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p

def _normalize(weights: Sequence[float]) -> np.ndarray:
    arr = np.asarray(weights, dtype=float)
    arr = np.clip(arr, 0.0, np.inf)
    s = float(arr.sum())
    if s <= 0:
        # uniform fallback
        arr = np.ones_like(arr, dtype=float)
        s = float(arr.size)
    return (arr / s).astype(float)

def _build_indicator_state(weights: Sequence[float], indicator_mask: Sequence[bool]) -> tuple[QuantumCircuit, int]:
    """
    Prepare |psi> = sum_i sqrt(w_i) |i> on n data qubits, with an
    ancilla target qubit encoding the indicator f(i) into amplitude of |1>.
    Implementation note: we use initialize() for clarity—fine for ≤64 bins.
    """
    w = _normalize(weights)
    n_bins = len(w)
    dim = _nearest_pow2(n_bins)
    n_qubits = int(math.log2(dim))
    data = np.zeros(dim, dtype=complex)
    data[:n_bins] = np.sqrt(w[:n_bins])

    qc = QuantumCircuit(n_qubits + 1)  # data + ancilla
    # load amplitudes to data register
    qc.initialize(data, list(range(n_qubits)))

    # mark states where indicator is True; flip ancilla for those indices
    # (multi-controlled X with state preparation via classical loop—cheap for ≤64)
    for idx, flag in enumerate(indicator_mask[:n_bins]):
        if not flag:
            continue
        # Basis state |idx>
        # Build controls by x-ing zeros, then mcx to ancilla, then uncompute
        bits = [(idx >> k) & 1 for k in range(n_qubits)]
        # move to |111...> pattern
        for q, b in enumerate(bits):
            if b == 0:
                qc.x(q)
        qc.mcx(list(range(n_qubits)), n_qubits)  # ancilla is last
        # uncompute
        for q, b in enumerate(bits):
            if b == 0:
                qc.x(q)

    return qc, n_qubits  # ancilla index is n_qubits

def estimate_indicator_probability(
    weights: Sequence[float],
    indicator_mask: Sequence[bool],
    *,
    backend_target: Optional[str] = None,
    shots: int = 8192,
    ibm_instance: Optional[str] = None,
) -> float:
    """
    Estimate E[1_{indicator}] i.e., probability mass on bins where indicator=True.
    Defaults to Aer simulator. If environment sets PT_QUANT_TARGET='ibm' and IBM is configured,
    uses IBM Runtime Sampler with conservative shots.
    """
    qc, n_data = _build_indicator_state(weights, indicator_mask)
    anc = n_data  # last qubit

    target = backend_target or os.getenv("PT_QUANT_TARGET", "").strip().lower()
    if target == "ibm" and _IBM_AVAILABLE:
        try:
            # Requires IBM Cloud account and set QISKIT_IBM_TOKEN (or prior saved account)
            QiskitRuntimeService channel="ibm_quantum"  # type: ignore
            svc = QiskitRuntimeService() if ibm_instance is None else QiskitRuntimeService(instance=ibm_instance)
            sampler = IBMSampler(session=None, service=svc)
            res = sampler.run([qc], shots=min(max(1024, shots), 20000)).result()  # bound shots
            quasi = res[0].data.meas.get_counts()
        except Exception:
            # Fall back to Aer if IBM not available or rate-limited
            sampler = AerSampler(backend=Aer.get_backend("aer_simulator"))
            res = sampler.run([qc], shots=shots).result()
            quasi = res[0].data.meas.get_counts()
    else:
        sampler = AerSampler(backend=Aer.get_backend("aer_simulator"))
        res = sampler.run([qc], shots=shots).result()
        quasi = res[0].data.meas.get_counts()

    # We measured all qubits; ancilla is last bit in the outcome string.
    shots_total = 0
    shots_anc1 = 0
    for bitstr, cnt in quasi.items():
        shots_total += cnt
        if bitstr[-1] == "1":
            shots_anc1 += cnt

    if shots_total <= 0:
        return float(np.sum(_normalize(weights) * np.array(list(map(float, indicator_mask[:len(weights)])))))
    return shots_anc1 / float(shots_total)

def prob_up_from_terminal(terminal_prices: Sequence[float], s0: float, bins: int = 64) -> float:
    """
    Compress terminal prices into ≤bins histogram and estimate P(price > s0)
    using a quantum indicator circuit.
    """
    arr = np.asarray(terminal_prices, dtype=float)
    if arr.size == 0 or not np.isfinite(s0):
        return 0.5
    bins = int(max(8, min(64, bins)))
    hist, edges = np.histogram(arr, bins=bins)
    weights = hist.astype(float) + 1e-9  # avoid empties
    centers = 0.5 * (edges[:-1] + edges[1:])
    indicator = (centers > float(s0)).tolist()
    return float(estimate_indicator_probability(weights.tolist(), indicator))
