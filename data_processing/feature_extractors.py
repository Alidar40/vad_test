import numpy as np
import python_speech_features

from config import config

SAMPLE_RATE = config["sample_rate"]


def nofeature_extractor(frame):
    return frame


def _logfbank_extractor(frame, winlen, winstep, nfilt):
    logfbank = python_speech_features.logfbank(frame, SAMPLE_RATE, winlen=winlen, winstep=winstep, nfilt=nfilt)
    return logfbank[np.newaxis, ...]


def logfbank_8_extractor(frame):
    # return 8x8 feature map from 10ms frame
    return _logfbank_extractor(frame, winlen=0.003, winstep=0.001, nfilt=8)


def logfbank_32_extractor(frame):
    # return 32x32 feature map from 10ms frame
    return _logfbank_extractor(frame, winlen=0.0023, winstep=0.00025, nfilt=32)
