from difflib import SequenceMatcher

# threshold = 0.001 # Åbenbart den eneste threshold på hvor de faktisk er ens...


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def compare_audio(file1_name, file2_name):
    with open(file1_name, "rb") as file1, open(file2_name, "rb") as file2:
        file1_content = file1.read()
        file2_content = file2.read()

    sim_ratio = similar(file1_content, file2_content)

    print(sim_ratio)

compare_audio(r'C:\Users\nicol\Desktop\Code Projects\Python\P4-Audio\P4-Audio\Graphs - Schøler\InputRaw.png', r'C:\Users\nicol\Desktop\Code Projects\Python\P4-Audio\P4-Audio\Graphs - Schøler\ReconRaw.png')

# compare_audio(r'C:\Users\nicol\Desktop\Code Projects\Python\P4-Audio\P4-Audio\Graphs - Schøler\InputRawTrim.png', r'C:\Users\nicol\Desktop\Code Projects\Python\P4-Audio\P4-Audio\Graphs - Schøler\ReconRawTrim.png')
#
# compare_audio(r'C:\Users\nicol\Desktop\Code Projects\Python\P4-Audio\P4-Audio\Graphs - Schøler\InputRawZoom.png', r'C:\Users\nicol\Desktop\Code Projects\Python\P4-Audio\P4-Audio\Graphs - Schøler\ReconRawZoom.png')
#
# compare_audio(r'C:\Users\nicol\Desktop\Code Projects\Python\P4-Audio\P4-Audio\Graphs - Schøler\InputSpectogram.png', r'C:\Users\nicol\Desktop\Code Projects\Python\P4-Audio\P4-Audio\Graphs - Schøler\ReconSpectogram.png')
#
# compare_audio(r'C:\Users\nicol\Desktop\Code Projects\Python\P4-Audio\P4-Audio\Graphs - Schøler\InputMelSpectogram.png', r'C:\Users\nicol\Desktop\Code Projects\Python\P4-Audio\P4-Audio\Graphs - Schøler\ReconMelSpectogram.png')
