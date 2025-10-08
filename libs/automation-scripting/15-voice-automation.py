"""
15-voice-automation.py

Beginner's guide to voice/text-to-speech automation with pyttsx3.

Overview:
---------
pyttsx3 is a Python library for converting text to speech (TTS).
You can make your computer talk, create reminders, or build accessibility tools.

Main Features:
--------------
- Convert text to speech
- Change voice, rate, and volume
- Save speech to audio files

Examples:
---------
"""

import pyttsx3 # Import pyttsx3

engine = pyttsx3.init() # initialize pyttsx3 in a variable `for this example we use engine which is recommended`.

# 1. Say some words
words = "Python automation is fun."
engine.say(words) # Call the say() function and pass a string or a variable.
engine.runAndWait()

# 2. Change voice rate and volume
engine.setProperty('rate', 80)    # Speed of speech
engine.setProperty('volume', 0.5)  # Volume (0.0 to 1.0)
engine.say("This is a slower, quieter voice.")
engine.runAndWait()

# 3. List available voices and switch to an English male voice
voices = engine.getProperty('voices')
for v in voices: # type: ignore
    print("Switched to:", v.name)

    # print("Voice:", v.name, v.id)
    # # Try to find an English male voice
    # if "en" in v.languages[0] and "male" in v.name.lower():
    #     engine.setProperty('voice', v.id)
    #     print("Switched to:", v.name)

engine.say("This is an English male voice.")
engine.runAndWait()

# 4. Save speech to a file
# engine.save_to_file("Saving this to a file.", "output.mp3")
# engine.runAndWait()

print("Text-to-speech done.")

"""
Tips:
-----
- Install pyttsx3: pip install pyttsx3
- Works offline and cross-platform.
- Official docs: https://pyttsx3.readthedocs.io/en/latest/
- Useful for reminders, alarms, accessibility, and fun projects.
"""
