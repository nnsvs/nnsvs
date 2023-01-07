from copy import deepcopy

import numpy as np
import pysinsy
from nnmnkwii.io import hts
from nnmnkwii.preprocessing.f0 import interp1d
from nnsvs.io.hts import (
    full_to_mono,
    get_note_indices,
    label2phrases,
    label2phrases_str,
)
from nnsvs.svs import SPSVS


class NEUTRINO(SPSVS):
    """NEUTRINO-like interface for singig voice synthesis

    Args:
        model_dir (str): model directory
        device (str): device name
        verbose (int): verbose level
    """

    def __init__(self, model_dir, device="cpu", verbose=0):
        super().__init__(model_dir, device=device, verbose=verbose)

        if self.feature_type != "world":
            raise RuntimeError(f"Unsupported feature type: {self.feature_type}")
        if not self.config.get("use_world_codec", False):
            self.logger.warning(
                "WORLD coded is required to output NEUTRIN-compatible features"
            )

    def musicxml2label(self, input_file):
        """Convert musicXML to full and mono HTS labels

        Args:
            input_file (str): musicXML file
        """

        contexts = pysinsy.extract_fullcontext(input_file)
        full_labels = hts.HTSLabelFile.create_from_contexts(contexts)
        mono_labels = full_to_mono(full_labels)

        return full_labels, mono_labels

    def get_num_phrases(self, labels):
        """Get number of phrases

        Args:
            labels (nnmnkwii.io.hts.HTSLabelFile): HTS label

        Returns:
            int: number of phrases
        """
        phrases = label2phrases(labels)
        return len(phrases)

    def get_phraselist(self, full_labels, timing_labels):
        """Get phraselit from full and timing HTS labels

        Args:
            full_labels (nnmnkwii.io.hts.HTSLabelFile): full HTS label
            timing_labels (nnmnkwii.io.hts.HTSLabelFile): timing HTS label

        Returns:
            str: phraselist
        """
        note_indices = get_note_indices(full_labels)
        phraselist = label2phrases_str(timing_labels, note_indices)
        return phraselist

    def predict_acoustic(
        self,
        full_labels,
        timing_labels=None,
        style_shift=0,
        phrase_num=-1,
        trajectory_smoothing=True,
        trajectory_smoothing_cutoff=50,
        trajectory_smoothing_cutoff_f0=20,
        vuv_threshold=0.5,
        force_fix_vuv=False,
        fill_silence_to_rest=False,
    ):
        """Main inference of timing and acoustic predictions

        Args:
            full_labels (nnmnkwii.io.hts.HTSLabelFile): full HTS label
            timing_labels (nnmnkwii.io.hts.HTSLabelFile): timing HTS label
            style_shift (float): style shift parameter
            phrase_num (int): phrase number to use for inference
            trajectory_smoothing (bool): whether to apply trajectory smoothing
            trajectory_smoothing_cutoff (float): cutoff frequency for trajectory smoothing
            trajectory_smoothing_cutoff_f0 (float): cutoff frequency for trajectory
                smoothing for f0
            vuv_threshold (float): V/UV threshold
            force_fix_vuv (bool): whether to force fix V/UV
            fill_silence_to_rest (bool): Fill silence to rest frames.

        Returns:
            tuple: (f0, mgc, bap)
        """
        if timing_labels is None:
            self.logger.warning("'timing_labels' is not provided.")
            # Run timing prediction
            duration_modified_full_labels = self.predict_timing(full_labels)
            timing_labels = full_to_mono(duration_modified_full_labels)
        else:
            # Load pre-estimated timing
            duration_modified_full_labels = deepcopy(full_labels)
            duration_modified_full_labels.start_times = timing_labels.start_times.copy()
            duration_modified_full_labels.end_times = timing_labels.end_times.copy()

        if phrase_num >= 0:
            phrases = label2phrases(duration_modified_full_labels)
            if phrase_num > len(phrases):
                raise RuntimeError(
                    f"phrase_num is too large: {phrase_num} > {len(phrases)}"
                )

            # Use the specified phrase for inference
            duration_modified_full_labels = phrases[phrase_num]
            self.logger.info(f"Using phrase {phrase_num}/{len(phrases)} for inference")

        # Predict acoustic features
        # NOTE: if non-zero pre_f0_shift_in_cent is specified, the input pitch
        # will be shifted before running the acoustic model
        acoustic_features = super().predict_acoustic(
            duration_modified_full_labels,
            f0_shift_in_cent=style_shift * 100,
        )

        # Post-processing for acoustic features
        # NOTE: if non-zero post_f0_shift_in_cent is specified, the output pitch
        # will be shifted as a part of post-processing
        multistream_features = self.postprocess_acoustic(
            acoustic_features=acoustic_features,
            duration_modified_labels=duration_modified_full_labels,
            trajectory_smoothing=trajectory_smoothing,
            trajectory_smoothing_cutoff=trajectory_smoothing_cutoff,
            trajectory_smoothing_cutoff_f0=trajectory_smoothing_cutoff_f0,
            vuv_threshold=vuv_threshold,
            force_fix_vuv=force_fix_vuv,
            fill_silence_to_rest=fill_silence_to_rest,
            f0_shift_in_cent=-style_shift * 100,
        )
        assert len(multistream_features) == 4
        mgc, lf0, vuv, bap = multistream_features

        # Convert lf0 to f0
        f0 = np.exp(lf0.copy())
        f0[vuv < vuv_threshold] = 0

        # Make sure to have correct array layout and dtype
        # These parameters can be used to generate waveform by WORLD
        f0 = np.ascontiguousarray(f0).astype(np.float64)
        mgc = np.ascontiguousarray(mgc).astype(np.float64)
        bap = np.ascontiguousarray(bap).astype(np.float64)
        return f0, mgc, bap

    def predict_waveform(
        self,
        f0,
        mgc,
        bap,
        vocoder_type="world",
        vuv_threshold=0.5,
        dtype=np.int16,
        peak_norm=False,
        loudness_norm=False,
        target_loudness=-20,
    ):
        """Generate waveform from acoustic features

        Args:
            f0 (ndarray): f0
            mgc (ndarray): mel-cepstrum
            bap (ndarray): band-aperiodicity
            vocoder_type (str): vocoder type
            vuv_threshold (float): V/UV threshold
            dtype (np.dtype): Data type of the output waveform.
            peak_norm (bool): Whether to normalize the waveform by peak value.
            loudness_norm (bool): Whether to normalize the waveform by loudness.
            target_loudness (float): Target loudness in dB.

        Returns:
            ndarray: waveform
        """
        # Convert NEUTRINO-like features to NNSVS's one
        # (f0, mgc, bap) -> (mgc, lf0, vuv, bap)
        vuv = (f0 > 0).astype(np.float64).reshape(-1, 1)
        lf0 = f0.copy()
        lf0[np.nonzero(lf0)] = np.log(f0[np.nonzero(lf0)])
        lf0 = interp1d(lf0, kind="slinear")
        multistream_features = (mgc, lf0, vuv, bap)

        wav = super().predict_waveform(
            multistream_features=multistream_features,
            vocoder_type=vocoder_type,
            vuv_threshold=vuv_threshold,
        )
        wav = self.postprocess_waveform(
            wav,
            dtype=dtype,
            peak_norm=peak_norm,
            loudness_norm=loudness_norm,
            target_loudness=target_loudness,
        )

        return wav
