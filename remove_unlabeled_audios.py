import os
import json


audio_20_folder = "data/sons_al_balco_audios/audios_2020"
audio_21_folder = "data/sons_al_balco_audios/audios_2021"

audio_20_list = os.listdir(audio_20_folder)
audio_21_list = os.listdir(audio_21_folder)

lab_20_path = "data/sons_al_balco_labels/SAB-AudioTagging-2020.json"
lab_21_path = "data/sons_al_balco_labels/SAB-AudioTagging-2021.json"

with open(lab_20_path) as f:
    lab_20 = json.load(f)
with open(lab_21_path) as f:
    lab_21 = json.load(f)

lab_20_names = [e["original_filename"] for e in lab_20]
lab_21_names = [e["original_filename"] for e in lab_21]

print(f"2020 audios: {len(audio_20_list)}, labels: {len(lab_20_names)}")
print(f"2021 audios: {len(audio_21_list)}, labels: {len(lab_21_names)}")

print(len(set(audio_20_list).intersection(set(lab_20_names))))
print(len(set(audio_21_list).intersection(set(lab_21_names))))

for el in os.listdir(audio_20_folder):
    if el not in set(audio_20_list).intersection(set(lab_20_names)):
        print("removing ", el)
        os.remove(os.path.join(audio_20_folder, el))
print("end removing file 20")

for el in os.listdir(audio_21_folder):
    if el not in set(audio_21_list).intersection(set(lab_21_names)):
        print("removing ", el)
        os.remove(os.path.join(audio_21_folder, el))
print("end removing file 21")

for i, el in enumerate(lab_20):
    if el["original_filename"] not in audio_20_list:
        print("removing ", el["original_filename"])
        lab_20.pop(i)

with open(lab_20_path, "w") as f:
    json.dump(lab_20, f)
print("end removing label 20")

for i, el in enumerate(lab_21):
    if el["original_filename"] not in audio_21_list:
        lab_21.pop(i)
        print("removing ", el["original_filename"])

with open(lab_21_path, "w") as f:
    json.dump(lab_21, f)
print("end removing label 21")
