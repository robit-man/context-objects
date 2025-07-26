#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lms.py – Minimal LMS adaptive filter, with both batch and streaming APIs.

Exports
-------
- lms_filter_batch: run LMS over entire signals, optionally sweep µ.
- StreamingLMSFilter: stateful filter for real-time or block processing.
"""

import numpy as np
from typing import Optional, Sequence, Tuple, Union


def lms_filter_batch(
    desired_signal: np.ndarray,
    reference_input: np.ndarray,
    filter_coeff: np.ndarray,
    step_size: Union[float, Sequence[float]],
    *,
    num_iterations: Optional[int] = None,
    return_error: bool = True,
    safe: bool = False
) -> Tuple[Optional[np.ndarray], np.ndarray, float]:
    """
    Batch LMS adaptive filter.

    Parameters
    ----------
    desired_signal : np.ndarray, shape (N,)
        Target signal d[n].
    reference_input : np.ndarray, shape (N,)
        Echo/source signal u[n].
    filter_coeff : np.ndarray, shape (M,)
        Initial filter taps f[0].
    step_size : float or sequence of floats
        If a float: use single µ. If sequence: try each µ and pick
        the one yielding minimal total squared error.
    num_iterations : int, optional
        Number of samples to run (default = len(desired_signal)).
    return_error : bool
        If True, returns the full error signal e[n].
    safe : bool
        If True, clip each sample error to ±1e4 to prevent overflow.

    Returns
    -------
    err : np.ndarray or None
        Error signal array (or None if return_error=False).
    best_coeff : np.ndarray, shape (M,)
        Adapted filter taps.
    best_mu : float
        The µ actually used (or best µ if sweeping).
    """
    N = len(desired_signal)
    M = len(filter_coeff)
    num_iterations = N if num_iterations is None or num_iterations > N else num_iterations

    # unify step_size to sequence
    mu_list = [step_size] if isinstance(step_size, float) else list(step_size)

    best_total_err = np.inf
    best_coeff = filter_coeff.copy()
    best_mu = mu_list[0]
    best_err = None

    for mu in mu_list:
        f = filter_coeff.astype(np.float64).copy()
        e = np.zeros(N, dtype=np.float32)

        for n in range(M, num_iterations):
            # build reversed window of reference_input[n-M+1 : n+1]
            u_block = reference_input[n : n - M : -1]
            y = np.dot(f, u_block)
            err = desired_signal[n] - y
            if safe:
                err = np.clip(err, -1e4, 1e4)
            e[n] = err
            f += mu * err * u_block  # f stays float64

        total_err = float(np.sum(e.astype(np.float64)**2))
        if total_err < best_total_err:
            best_total_err = total_err
            best_mu = mu
            best_coeff = f.astype(filter_coeff.dtype)
            best_err = e.copy()

    return (best_err if return_error else None), best_coeff, best_mu


class StreamingLMSFilter:
    """
    Stateful LMS filter for streaming or block processing.

    Attributes
    ----------
    coeffs : np.ndarray, shape (M,)
        Current filter taps.
    mu : float
        Step size.
    safe : bool
        If True, clip each sample error to ±1e4.
    """

    def __init__(self, num_taps: int, mu: float, safe: bool = False) -> None:
        """
        Parameters
        ----------
        num_taps : int
            Number of filter taps (M).
        mu : float
            Step size.
        safe : bool
            Enable overflow-safe clipping.
        """
        self.coeffs = np.zeros(num_taps, dtype=np.float32)
        self.mu = mu
        self.safe = safe
        # circular buffer to hold last M reference samples
        self._buffer = np.zeros(num_taps, dtype=np.float32)

    def process_block(
        self,
        reference_block: np.ndarray,
        desired_block: np.ndarray
    ) -> np.ndarray:
        """
        Process a block of samples in streaming mode.

        Parameters
        ----------
        reference_block : np.ndarray, shape (L,)
            New reference samples u[n].
        desired_block : np.ndarray, shape (L,)
            New desired samples d[n].

        Returns
        -------
        error_block : np.ndarray, shape (L,)
            Filter output error e[n] = d[n] – y[n].
        """
        L = len(desired_block)
        error_block = np.zeros(L, dtype=np.float32)

        for i in range(L):
            # shift in newest reference sample
            self._buffer = np.roll(self._buffer, 1)
            self._buffer[0] = reference_block[i]

            # filter output
            y = float(np.dot(self.coeffs, self._buffer))
            e = float(desired_block[i] - y)
            if self.safe:
                e = np.clip(e, -1e4, 1e4)
            error_block[i] = e

            # update taps
            self.coeffs += self.mu * e * self._buffer

        return error_block

    def process_sample(self, ref_sample: float, des_sample: float) -> float:
        """
        Process a single sample (u[n], d[n]) and update state.

        Returns
        -------
        e : float
            The error sample.
        """
        # reuse block logic for single sample
        return float(self.process_block(
            np.array([ref_sample], dtype=np.float32),
            np.array([des_sample], dtype=np.float32)
        )[0])


# ─── Demo when run as script ────────────────────────────────────────────────
if __name__ == "__main__":
    print("▶ Running LMS demo…")

    # synthetic signals
    rng = np.random.default_rng(0)
    N = 5000
    clean = rng.standard_normal(N).astype(np.float32)
    echo = np.concatenate((np.zeros(50), 0.6 * clean[:-50]))

    # make an echoy desired signal
    desired = clean + echo
    reference = desired  # in practice, capture speaker output here

    # Batch mode: sweep µ
    init_coeffs = np.zeros(128, dtype=np.float32)
    _, coeffs, mu = lms_filter_batch(
        desired, reference, init_coeffs, [1e-4, 5e-4, 1e-3], safe=True
    )
    print(f"Batch mode → best µ: {mu:.1e}")

    # Streaming mode: fixed µ
    filt = StreamingLMSFilter(num_taps=128, mu=5e-4, safe=True)
    err_block = filt.process_block(reference, desired)
    print(f"Streaming mode → final total error: {np.sum(err_block**2):.2f}")
