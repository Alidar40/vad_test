import numpy as np
import matplotlib.pyplot as plt

from config import config

SAMPLE_RATE = config['sample_rate']


def _norm_audio(audio: np.array):
    '''
    Normalize the raw signal into a [0..1] range.
    '''
    return audio / np.max(np.abs(audio), axis=0)


def _time_axis(raw, labels):
    '''
    Generates time axis for a raw signal and its labels.
    '''
    time = np.linspace(0, len(raw) / SAMPLE_RATE, num=len(raw))
    time_labels = np.linspace(0, len(raw) / SAMPLE_RATE, num=len(labels))
    return time, time_labels


def plot_audio_with_vad(audio: np.array, predicted_labels: np.array, true_labels: np.array, title="", normalize=False):
    '''
    Plot a raw signal as waveform and its corresponding labels.
    '''
    predicted_labels = predicted_labels.flatten()
    true_labels = true_labels.flatten()
    scaler = np.ones_like(true_labels)
    if normalize:
        audio = _norm_audio(audio.flatten())
    time, time_labels = _time_axis(audio, true_labels)

    fig = plt.figure(1, figsize=(16, 3))
    plt.plot(time, audio)
    plt.plot(time_labels, scaler, color='#E5ECF6')
    plt.plot(time_labels, scaler*-1, color='#E5ECF6')
    plt.plot(time_labels, predicted_labels, color='red', label="Predicted")
    plt.plot(time_labels, true_labels, color='black', label="Truth")
    plt.title(title)
    plt.legend()
    return fig


def plot_roc(fpr, tpr, roc_auc=None, title="Receiver Operating Characteristic"):
    '''
    Plot ROC-curve
    '''
    plt.title(title)
    if roc_auc:
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    else:
        plt.plot(fpr, tpr, 'b')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
