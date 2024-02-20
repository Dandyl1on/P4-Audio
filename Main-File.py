from pydub import AudioSegment
from pydub.playback import play

sound = AudioSegment.from_wav("GI_GMF_B3_353_20140520_n.wav")
play(sound)