def detect_profanity(text):
    profanity_list = ["badword1", "badword2", "badword3"]  # Add more words as needed
    detected_profanity = [word for word in profanity_list if word in text.lower()]
    return detected_profanity

def is_profane(text):
    return len(detect_profanity(text)) > 0

if __name__ == "__main__":
    sample_text = "This is a sample text with badword1."
    if is_profane(sample_text):
        print("Profanity detected!")
    else:
        print("No profanity detected.")