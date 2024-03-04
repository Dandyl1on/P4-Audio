import wave
import pyaudio

# Open the WAV file
with wave.open('../GI_GMF_B3_353_20140520_n.wav', 'rb') as wf:
    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open a stream
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                     channels=wf.getnchannels(),  # Use the number of channels from the WAV file
                     rate=wf.getframerate(),
                     output=True)

    # Read data
    data = wf.readframes(1024)

    # Play the audio
    while data:
        stream.write(data)
        data = wf.readframes(1024)

    # Close the stream and PyAudio
    stream.stop_stream()
    stream.close()
    p.terminate()
