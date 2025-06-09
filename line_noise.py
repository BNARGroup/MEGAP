# Copyright 2011-2024 MNE-Python authors

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# For more information, please check https://github.com/mne-tools/mne-python/blob/main/mne/time_frequency/multitaper.py

import mne 
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from collections import Counter
from copy import deepcopy
from functools import partial
import numpy as np
from scipy import fft, signal
from scipy.stats import f as fstat
from mne import filter
from mne.utils import logger, verbose, warn, _pl
from mne.parallel import parallel_func
from mne.utils import sum_squared, _check_preload, _validate_type, _check_option, _ensure_int
from mne.filter import _check_method, _prep_for_filtering, _to_samples, _get_window_thresh, _mt_spectrum_remove_win, _check_filterable, _get_window_thresh, _mt_spectrum_remove, _prep_for_filtering, _mt_spectrum_proc
from mne.io.base import BaseRaw
from mne.epochs import BaseEpochs
from mne import Evoked
from mne._ola import _COLA
from mne.time_frequency.multitaper import _mt_spectra
import os
from initial_files import initial_files
from initial_py import read_cfg,redirect_stdout,find_meg_device


def _mt_spectrum_remove_win_edited(
    x, sfreq, line_freqs, notch_widths, window_fun, threshold, get_thresh, min_freq, max_freq
):
    n_times = x.shape[-1]
    n_samples = window_fun.shape[1]
    n_overlap = (n_samples + 1) // 2
    x_out = np.zeros_like(x)
    rm_freqs = list()
    idx = [0]

    # Define how to process a chunk of data
    def process(x_):
        out = _mt_spectrum_remove_edited(
            x_, sfreq, line_freqs, notch_widths, window_fun, threshold, get_thresh, min_freq, max_freq
        )
        rm_freqs.append(out[1])
        return (out[0],)  # must return a tuple

    # Define how to store a chunk of fully processed data (it's trivial)
    def store(x_):
        stop = idx[0] + x_.shape[-1]
        x_out[..., idx[0]: stop] += x_
        idx[0] = stop

    _COLA(process, store, n_times, n_samples,
          n_overlap, sfreq, verbose=False).feed(x)
    assert idx[0] == n_times
    return x_out, rm_freqs


def _mt_spectrum_remove_edited(
    x, sfreq, line_freqs, notch_widths, window_fun, threshold, get_thresh, min_freq, max_freq
):
    """Use MT-spectrum to remove line frequencies.

    Based on Chronux. If line_freqs is specified, all freqs within notch_width
    of each line_freq is set to zero.
    """

    assert x.ndim == 1
    if x.shape[-1] != window_fun.shape[-1]:
        window_fun, threshold = get_thresh(x.shape[-1])
    # drop the even tapers
    n_tapers = len(window_fun)
    tapers_odd = np.arange(0, n_tapers, 2)
    tapers_even = np.arange(1, n_tapers, 2)
    tapers_use = window_fun[tapers_odd]

    # sum tapers for (used) odd prolates across time (n_tapers, 1)
    H0 = np.sum(tapers_use, axis=1)

    # sum of squares across tapers (1, )
    H0_sq = sum_squared(H0)

    # make "time" vector
    rads = 2 * np.pi * (np.arange(x.size) / float(sfreq))

    # compute mt_spectrum (returning n_ch, n_tapers, n_freq)
    x_p, freqs = _mt_spectra(x[np.newaxis, :], window_fun, sfreq)

    # sum of the product of x_p and H0 across tapers (1, n_freqs)
    x_p_H0 = np.sum(x_p[:, tapers_odd, :] *
                    H0[np.newaxis, :, np.newaxis], axis=1)

    # resulting calculated amplitudes for all freqs
    A = x_p_H0 / H0_sq

    if line_freqs is None:
        # figure out which freqs to remove using F stat

        # estimated coefficient
        x_hat = A * H0[:, np.newaxis]

        # numerator for F-statistic
        num = (n_tapers - 1) * (A * A.conj()).real * H0_sq
        # denominator for F-statistic
        den = np.sum(np.abs(x_p[:, tapers_odd, :] - x_hat) ** 2, 1) + np.sum(
            np.abs(x_p[:, tapers_even, :]) ** 2, 1
        )
        den[den == 0] = np.inf
        f_stat = num / den

        # find frequencies to remove

        indices = np.where(f_stat > threshold)[1]

        rm_freqs = freqs[indices]

        if any(freq < min_freq for freq in rm_freqs) or any(freq >= max_freq for freq in rm_freqs):
            # Filter out frequencies lower than 250 Hz and greater than or equal to 45 Hz
            rm_freqs = [freq for freq in rm_freqs if freq >=
                        max_freq and freq < min_freq]

    else:
        # specify frequencies
        indices_1 = np.unique([np.argmin(np.abs(freqs - lf))
                              for lf in line_freqs])
        indices_2 = [
            np.logical_and(freqs > lf - nw / 2.0, freqs < lf + nw / 2.0)
            for lf, nw in zip(line_freqs, notch_widths)
        ]
        indices_2 = np.where(np.any(np.array(indices_2), axis=0))[0]
        indices = np.unique(np.r_[indices_1, indices_2])

        rm_freqs = freqs[indices]

    fits = list()
    for ind in indices:
        c = 2 * A[0, ind]
        fit = np.abs(c) * np.cos(freqs[ind] * rads + np.angle(c))
        fits.append(fit)

    if len(fits) == 0:
        datafit = 0.0
    else:
        # fitted sinusoids are summed, and subtracted from data
        datafit = np.sum(fits, axis=0)

    return x - datafit, rm_freqs


def notch_filter_edited(
    x,
    Fs,
    freqs,
    filter_length="auto",
    notch_widths=None,
    trans_bandwidth=1,
    method="fir",
    iir_params=None,
    mt_bandwidth=None,
    p_value=0.05,
    picks=None,
    n_jobs=None,
    copy=True,
    phase="zero",
    fir_window="hamming",
    fir_design="firwin",
    pad="reflect_limited",
    min_freq=None,
    max_freq=None,
    *,
    verbose=None,
):

    x = _check_filterable(x, "notch filtered", "notch_filter")
    iir_params, method = _check_method(method, iir_params, ["spectrum_fit"])
    if min_freq is None:
        min_freq = 0
    if max_freq is None:
        max_freq = Fs/2
    if freqs is not None:
        freqs = np.atleast_1d(freqs)
    elif method != "spectrum_fit":
        raise ValueError(
            "freqs=None can only be used with method " "spectrum_fit")

    # Only have to deal with notch_widths for non-autodetect
    if freqs is not None:
        if notch_widths is None:
            notch_widths = freqs / 200.0
        elif np.any(notch_widths < 0):
            raise ValueError("notch_widths must be >= 0")
        else:
            notch_widths = np.atleast_1d(notch_widths)
            if len(notch_widths) == 1:
                notch_widths = notch_widths[0] * np.ones_like(freqs)
            elif len(notch_widths) != len(freqs):
                raise ValueError(
                    "notch_widths must be None, scalar, or the " "same length as freqs"
                )

    if method in ("fir", "iir"):
        pass
    elif method == "spectrum_fit":
        xf, line_noise_freq, final_line = _mt_spectrum_proc_edited(
            x,
            Fs,
            freqs,
            notch_widths,
            mt_bandwidth,
            p_value,
            picks,
            n_jobs,
            copy,
            filter_length,
            min_freq,
            max_freq,

        )
    print(min_freq, max_freq)
    return xf, line_noise_freq, final_line


def _mt_spectrum_proc_edited(
    x,
    sfreq,
    line_freqs,
    notch_widths,
    mt_bandwidth,
    p_value,
    picks,
    n_jobs,
    copy,
    filter_length,
    min_freq=None,
    max_freq=None,

):
    """Call _mt_spectrum_remove with the minimum frequency."""
    if min_freq is None:
        min_freq = 0
    if max_freq is None:
        max_freq = sfreq/2
    # set up array for filtering, reshape to 2D, operate on last axis
    x, orig_shape, picks = _prep_for_filtering(x, copy, picks)
    if isinstance(filter_length, str) and filter_length == "auto":
        filter_length = "10s"
    if filter_length is None:
        filter_length = x.shape[-1]
    filter_length = min(_to_samples(filter_length, sfreq, "", ""), x.shape[-1])
    get_wt = partial(
        _get_window_thresh, sfreq=sfreq, mt_bandwidth=mt_bandwidth, p_value=p_value
    )
    window_fun, threshold = get_wt(filter_length)
    parallel, p_fun, n_jobs = parallel_func(
        _mt_spectrum_remove_win_edited, n_jobs)
    if n_jobs == 1:
        freq_list = list()
        for ii, x_ in enumerate(x):
            if ii in picks:
                x[ii], f = _mt_spectrum_remove_win_edited(
                    x_, sfreq, line_freqs, notch_widths, window_fun, threshold, get_wt, min_freq, max_freq
                )
                freq_list.append(f)
    else:
        data_new = parallel(
            p_fun(x_, sfreq, line_freqs, notch_widths,
                  window_fun, threshold, get_wt)
            for xi, x_ in enumerate(x)
            if xi in picks
        )
        freq_list = [d[1] for d in data_new]
        data_new = np.array([d[0] for d in data_new])
        x[picks, :] = data_new

    # report found frequencies, but do some sanitizing first by binning into
    # 1 Hz bins
    counts = Counter(
        sum((np.unique(np.round(ff)).tolist()
            for f in freq_list for ff in f), list())
    )
    kind = "Detected" if line_freqs is None else "Removed"
    found_freqs = (
        "\n".join(
            f"    {freq:6.2f} : " f"{counts[freq]:4d}"
            for freq in sorted(counts)
        )
        or "    None"
    )
    logger.info(f"{kind} notch frequencies (Hz):\n{found_freqs}")

    final_line = []
    line_noise_freq = counts
    for item in line_noise_freq:
        if line_noise_freq[item] > 5:
            final_line.append(item)
 
    x.shape = orig_shape

    final_line = sorted(final_line)
    return x, line_noise_freq, final_line

def multi_taper_removal(subject_ids):
    """
    Apply line noise removal to MEG data and save the cleaned data.

    Args:
        subject_ids (list): List of subject IDs.
    """
    main_location = str(os.getenv('MAINMEG'))
    MEG_device = str(os.getenv('MEGDEVICE'))
    find_meg_device()
    cfg = read_cfg(main_location)
    step = 'multi_taper_removal'
    cfg = cfg[step]

    for subject_id in subject_ids:
        # Create a log file for the subject
        log_file_path = os.path.join(main_location, 'verbose', f'{subject_id}.txt')
        with open(log_file_path, 'a') as log_file:
            with redirect_stdout(log_file, step):
                print(f"Config= {cfg}\n")

                # Load data and prepare folder
                data, _, folder_path = initial_files(subject_id, main_location, "multi_taper_removal", "zapline_plus")
                sfreq = data.info['sfreq']

                # Apply line noise removal to MEG channels 
                if MEG_device=="MEGIN":
                    meg_ch_idx = mne.pick_types(data.info, meg="grad")
                    grad_data, line_noise_freq, final_line = notch_filter_edited(data.get_data(picks=meg_ch_idx), sfreq, **cfg)
                    data._data[meg_ch_idx] = grad_data

                meg_ch_idx = mne.pick_types(data.info, meg="mag",ref_meg=True)
                mag_data, line_noise_freq, final_line = notch_filter_edited(data.get_data(picks=meg_ch_idx), sfreq, **cfg)
                data._data[meg_ch_idx] = mag_data

                # Save the cleaned data
                data.save(os.path.join(folder_path, f"{subject_id}.fif"), overwrite=True, fmt='double')

                