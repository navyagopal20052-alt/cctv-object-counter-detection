import wave
import math
import struct
import os

# path where file must be created
TARGET_PATH = r"C:\project\frontend\assets\audio\alert_bell.wav"

# 4 seconds bell sound
duration = 4.0
sample_rate = 44100
frequency = 880  # bell type frequency

total_samples = int(duration * sample_rate)

# ensure folder exists
folder = os.path.dirname(TARGET_PATH)
if not os.path.exists(folder):
    print("ERROR: FOLDER NOT FOUND:", folder)
    exit()

# generate sound and save
with wave.open(TARGET_PATH, "w") as wav:
    wav.setparams((1, 2, sample_rate, total_samples, "NONE", "not compressed"))

    for i in range(total_samples):
        amplitude = int(32767.0 * math.sin(2 * math.pi * frequency * i / sample_rate))
        wav.writeframes(struct.pack('<h', amplitude))

# show success
size_kb = os.path.getsize(TARGET_PATH) / 1024.0
print("✅ SUCCESS: File created:", TARGET_PATH)
print(f"size = {size_kb:.2f} KB")
