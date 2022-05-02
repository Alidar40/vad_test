from abc import ABC

import torch
import webrtcvad
import pvcobra

from config import config


FRAME_SIZE = config["frame_size"]
SAMPLE_RATE = config["sample_rate"]
COBRA_ACCESS_KEY = config["cobra_access_key"]
COBRA_THRESHOLD = config["cobra_threshold"]


class ExternalVAD(ABC):
    def __init__(self):
        pass

    def __call__(self, batch):
        pass


class WebrtcVAD(ExternalVAD):
    def __init__(self):
        super(WebrtcVAD, self).__init__()

    def __call__(self, batch):
        # Usually it works far better if you reinit the VAD every time you call it
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(3)

        if type(batch) == torch.Tensor:
            batch = batch.detach().cpu().numpy()
            predictions = list()
            for frame in batch:
                # why 'not is_speech'? Good question
                predictions.append(not self.vad.is_speech(frame.tobytes(), SAMPLE_RATE))

            return torch.unsqueeze(torch.as_tensor(predictions, dtype=torch.float), dim=1)

        return not self.vad.is_speech(batch.tobytes(), SAMPLE_RATE)


class CobraVAD(ExternalVAD):
    def __init__(self):
        super(CobraVAD, self).__init__()
        self.vad = pvcobra.create(access_key=COBRA_ACCESS_KEY)
        # self.vad.sample_rate = self.sample_rate
        self.vad._frame_length = FRAME_SIZE
        self.threshold = COBRA_THRESHOLD

    def __call__(self, batch):
        if type(batch) == torch.Tensor:
            batch = batch.detach().cpu().numpy()
            predictions = list()
            for frame in batch:
                predictions.append(self.vad.process(frame))

            return torch.unsqueeze(torch.as_tensor(predictions, dtype=torch.int), dim=1)

        if self.vad.process(batch) > self.threshold:
            return True
        return False
