from abc import ABC

import webrtcvad
import pvcobra


class VAD_Classifier(ABC):
    def __init__(self, frame_size, sample_rate):
        self.frame_size = frame_size
        self.sample_rate = sample_rate

    def is_speech(self, frame):
        pass

    def reinit(self):
        pass


class WebrtcVAD(VAD_Classifier):
    def __init__(self, frame_size, sample_rate):
        super(WebrtcVAD, self).__init__(frame_size, sample_rate)
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(3)

    def is_speech(self, frame):
        # why 'not is_speech'? I don't know
        return not self.vad.is_speech(frame.tobytes(), self.sample_rate)

    def reinit(self):
        # Usually it works far better if you reinit the VAD every time you call it
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(3)


class CobraVAD(VAD_Classifier):
    def __init__(self, frame_size, sample_rate, threshold, access_key):
        super(CobraVAD, self).__init__(frame_size, sample_rate)
        self.vad = pvcobra.create(access_key=access_key)
        # self.vad.sample_rate = self.sample_rate
        self.vad._frame_length = self.frame_size
        self.threshold = threshold

    def is_speech(self, frame):
        prob = self.vad.process(frame)
        print(prob)
        if prob > self.threshold:
            return True
        return False
