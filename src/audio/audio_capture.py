import pyaudio
import wave

class AudioCapture:
    def __init__(self, channels=1, rate=44100, chunk=1024):
        self.channels = channels
        self.rate = rate
        self.chunk = chunk
        self.p = pyaudio.PyAudio()
        self.stream = None

    def start_stream(self):
        self.stream = self.p.open(format=pyaudio.paInt16,
                                   channels=self.channels,
                                   rate=self.rate,
                                   input=True,
                                   frames_per_buffer=self.chunk)

    def stop_stream(self):
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

    def record(self, duration):
        frames = []
        self.start_stream()
        for _ in range(0, int(self.rate / self.chunk * duration)):
            data = self.stream.read(self.chunk)
            frames.append(data)
        self.stop_stream()
        return frames

    def save_recording(self, frames, filename):
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(frames))

    def __del__(self):
        if self.stream is not None:
            self.stop_stream()
        self.p.terminate()