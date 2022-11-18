from datasets import load_dataset


langs = ["en", "de", "es", "fr", "hi", "th"]
splits = ["eval", "test", "train"]
split_dict = {}
for lang in langs:
    for split in splits:
        split_dict[f"{split}_{lang}"] = f"./mtop/{lang}/{split}.txt.csv"

d = load_dataset("csv", data_files=split_dict)
d.push_to_hub("WillHeld/mtop")

splits = ["validation", "test", "train"]
split_dict = {}
for split in splits:
    split_dict[f"{split}"] = f"./cstop/{split}.tsv.csv"

d = load_dataset("csv", data_files=split_dict)
d.push_to_hub("WillHeld/hinglish_top")

splits = ["eval", "test", "train"]
domains = [
    "alarm",
    "event",
    "messaging",
    "music",
    "navigation",
    "reminder",
    "timer",
    "weather",
]
split_dict = {}
for split in splits:
    split_dict[split] = [f"./topv2/{domain}_{split}.tsv.csv" for domain in domains]
d = load_dataset("csv", data_files=split_dict)
d.push_to_hub("WillHeld/top_v2")

splits = ["eval", "test", "train"]
split_dict = {}
for split in splits:
    split_dict[split] = f"./CSTOP/CSTOP_{split}.tsv.csv"
d = load_dataset("csv", data_files=split_dict)
d.push_to_hub("WillHeld/cstop")