import wave
import numpy as np

def read_wav_file(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        # Get the number of channels (1 for mono, 2 for stereo)
        num_channels = wav_file.getnchannels()
        
        # Get the sample width in bytes (e.g., 2 for 16-bit samples)
        sample_width = wav_file.getsampwidth()
        
        # Get the sample rate (samples per second)
        sample_rate = wav_file.getframerate()
        
        # Get the number of frames (samples)
        num_frames = wav_file.getnframes()
        
        # Read the audio data from the WAV file
        audio_data = wav_file.readframes(num_frames)
        
        # Convert the binary audio data to a NumPy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16 if sample_width == 2 else np.int8)
        
        return audio_array, num_channels, sample_width, sample_rate