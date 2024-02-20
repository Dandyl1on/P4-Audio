from difflib import SequenceMatcher

threshold = 0.001


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def compare_audio(file1_name, file2_name):
    with open(file1_name, "rb") as file1, open(file2_name, "rb") as file2:
        file1_content = file1.read()
        file2_content = file2.read()

    sim_ratio = similar(file1_content, file2_content)

    if sim_ratio > threshold:
        print('Same')
    else:
        print('Different')

compare_audio('GI_GMF_B3_353_20140520_n.wav', 'reconstructed_audio.wav')
