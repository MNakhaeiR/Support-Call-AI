from speech_recognition import Recognizer, Microphone

class SpeechToText:
    def __init__(self):
        self.recognizer = Recognizer()

    def recognize_speech(self):
        with Microphone() as source:
            print("Adjusting for ambient noise... Please wait.")
            self.recognizer.adjust_for_ambient_noise(source)
            print("Listening...")
            audio = self.recognizer.listen(source)

        try:
            print("Recognizing...")
            text = self.recognizer.recognize_google(audio)
            print("You said: " + text)
            return text
        except Exception as e:
            print("Could not understand audio, error: " + str(e))
            return None

if __name__ == "__main__":
    stt = SpeechToText()
    stt.recognize_speech()