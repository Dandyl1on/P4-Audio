from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def compare_audio(file1_name, file2_name):
    with open(file1_name, "rb") as file1, open(file2_name, "rb") as file2:
        file1_content = file1.read()
        file2_content = file2.read()

    sim_ratio = similar(file1_content, file2_content)

    print(f"Similarity ratio between {file1_name} and {file2_name}: {sim_ratio}")

# Glossary
# Input = Original Audio file
# Recon = Inverse of fourier transform
#


directory = r'C:\Users\nicol\Desktop\Code Projects\Python\P4-Audio\P4-Audio\Graphs - Schøler'
files = [
    "InputRaw.png",
    "InputRawTrim.png",
    "InputRawZoom.png",
    "InputSpectogram.png",
    "InputMelSpectogram.png",
    "ReconRaw.png",
    "ReconRawTrim.png",
    "ReconRawZoom.png",
    "ReconSpectogram.png",
    "ReconMelSpectogram.png"
]

for i in range(0, len(files), 2):
    file1_name = os.path.join(directory, files[i])
    file2_name = os.path.join(directory, files[i + 1])
    compare_audio(file1_name, file2_name)