from models.naive_linear_net import NaiveLinearNet
from models.lenet import LeNet8, LeNet32
from models.cnn_lstm import CNN_BiLSTM
from models.external_vad import WebrtcVAD, CobraVAD
from data_processing.feature_extractors import nofeature_extractor, logfbank_8_extractor, logfbank_32_extractor


def get_classifier_by_name(model_name):
    if model_name == "naive_linear":
        classifier = NaiveLinearNet()
        feature_extractor = nofeature_extractor
    elif model_name == "webrtc":
        # For test/benchmarking only!
        classifier = WebrtcVAD()
        feature_extractor = nofeature_extractor
    elif model_name == "cobra":
        # For test/benchmarking only!
        classifier = CobraVAD()
        feature_extractor = nofeature_extractor
    elif model_name == "lenet8":
        classifier = LeNet8()
        feature_extractor = logfbank_8_extractor
    elif model_name == "lenet32":
        classifier = LeNet32()
        feature_extractor = logfbank_32_extractor
    elif model_name == "cnn_bilstm":
        classifier = CNN_BiLSTM()
        feature_extractor = logfbank_32_extractor
    else:
        raise NotImplemented("Such a model is not implemented")

    return classifier, feature_extractor


def get_vad_by_name(vad_name):
    if vad_name == "webrtc":
        vad = WebrtcVAD()
    elif vad_name == "cobra":
        vad = CobraVAD()
    else:
        raise NotImplemented("Such a VAD classifier is not implemented")

    return vad
