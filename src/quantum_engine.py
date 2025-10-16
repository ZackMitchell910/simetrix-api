from __future__ import annotations

import math
import os
import logging
from typing import Optional, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer.primitives import Sampler as AerSampler

try:  # Optional IBM Runtime (enabled when configured)
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as IBMSampler
    _IBM_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency may be missing
    QiskitRuntimeService = None  # type: ignore
    IBMSampler = None  # type: ignore
    _IBM_AVAILABLE = False

logger = logging.getLogger(__name__)


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
        arr = np.ones_like(arr, dtype=float)
        s = float(arr.size)
    return (arr / s).astype(float)


def _build_indicator_state(
    weights: Sequence[float],
    indicator_mask: Sequence[bool],
) -> tuple[QuantumCircuit, int]:
    """
    Prepare |psi> = sum_i sqrt(w_i) |i> on n data qubits, with an
    ancilla target qubit encoding the indicator f(i) into amplitude of |1>.
    Implementation uses initialize() which is acceptable for <=64 bins.
    """
    w = _normalize(weights)
    n_bins = len(w)
    dim = _nearest_pow2(n_bins)
    n_qubits = int(math.log2(dim))
    data = np.zeros(dim, dtype=complex)
    data[:n_bins] = np.sqrt(w[:n_bins])

    qc = QuantumCircuit(n_qubits + 1)  # data + ancilla
    qc.initialize(data, list(range(n_qubits)))

    for idx, flag in enumerate(indicator_mask[:n_bins]):
        if not flag:
            continue
        bits = [(idx >> k) & 1 for k in range(n_qubits)]
        for q, b in enumerate(bits):
            if b == 0:
                qc.x(q)
        qc.mcx(list(range(n_qubits)), n_qubits)
        for q, b in enumerate(bits):
            if b == 0:
                qc.x(q)

    return qc, n_qubits


def _build_ibm_sampler(
    *,
    instance: Optional[str] = None,
    backend: Optional[str] = None,
) -> Optional[IBMSampler]:
    if not _IBM_AVAILABLE:
        return None
    try:
        svc_kwargs: dict[str, str] = {}
        token = os.getenv("QISKIT_IBM_TOKEN", "").strip()
        channel = os.getenv("QISKIT_IBM_CHANNEL", "ibm_quantum").strip()
        if token:
            svc_kwargs["token"] = token
        if channel:
            svc_kwargs["channel"] = channel
        inst = instance or os.getenv("QISKIT_IBM_INSTANCE", "").strip()
        if inst:
            svc_kwargs["instance"] = inst
        svc = QiskitRuntimeService(**svc_kwargs)  # type: ignore[arg-type]
        return IBMSampler(service=svc, backend=backend or os.getenv("PT_QUANT_IBM_BACKEND") or None)
    except Exception as exc:  # pragma: no cover - external service setup
        logger.warning("Unable to initialise IBM Runtime sampler: %s", exc)
        return None


def estimate_indicator_probability(
    weights: Sequence[float],
    indicator_mask: Sequence[bool],
    *,
    backend_target: Optional[str] = None,
    shots: int = 8192,
    ibm_instance: Optional[str] = None,
    ibm_backend: Optional[str] = None,
) -> float:
    """
    Estimate E[1_{indicator}] i.e., probability mass where indicator=True.
    Defaults to Aer; if PT_QUANT_TARGET=ibm and IBM is configured, uses Runtime.
    """
    qc, n_data = _build_indicator_state(weights, indicator_mask)
    target = (backend_target or os.getenv("PT_QUANT_TARGET", "")).strip().lower()
    use_ibm = target in {"ibm", "ibmq", "ibm_runtime"}
    quasi: dict[str, int] = {}

    if use_ibm:
        sampler = _build_ibm_sampler(instance=ibm_instance, backend=ibm_backend)
        if sampler:
            try:
                result = sampler.run(
                    [qc],
                    shots=min(max(1024, shots), 20000),
                ).result()
                quasi = result[0].data.meas.get_counts()
            except Exception as exc:  # pragma: no cover - external service
                logger.warning("IBM sampler run failed, reverting to Aer: %s", exc)
                quasi = {}
                use_ibm = False
        else:
            use_ibm = False

    if not use_ibm:
        sampler = AerSampler()
        result = sampler.run([qc], shots=shots).result()
        quasi = result[0].data.meas.get_counts()

    shots_total = 0
    shots_anc1 = 0
    for bitstr, cnt in quasi.items():
        shots_total += cnt
        if bitstr[-1] == "1":
            shots_anc1 += cnt

    if shots_total <= 0:  # fallback to classical estimate
        probs = _normalize(weights)
        indicator = np.asarray(indicator_mask[: len(weights)], dtype=float)
        return float(np.dot(probs, indicator))
    return shots_anc1 / float(shots_total)


def prob_up_from_terminal(
    terminal_prices: Sequence[float],
    s0: float,
    bins: int = 64,
) -> float:
    """
    Compress terminal price distribution into bins and estimate P(price > s0)
    with a quantum amplitude-estimation style indicator circuit.
    """
    arr = np.asarray(terminal_prices, dtype=float)
    if arr.size == 0 or not np.isfinite(s0):
        return 0.5
    bins = int(max(8, min(64, bins)))
    hist, edges = np.histogram(arr, bins=bins)
    weights = hist.astype(float) + 1e-9
    centers = 0.5 * (edges[:-1] + edges[1:])
    indicator = (centers > float(s0)).tolist()
    return float(estimate_indicator_probability(weights.tolist(), indicator))
