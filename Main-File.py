import math
from pydub import AudioSegment
from pydub.playback import play

song = AudioSegment.from_file("GI_GMF_B3_353_20140520_n.wav")
play(song)

