from datasets import load_dataset


parsing_name_mapping = {
    "mtop": ("utterance", "dcp_form", "locale"),
    "hinglish_top": ("cs_query", "cs_parse"),
    "top_v2": ("utterance", "semantic_parse"),
    "cstop": ("utterance", "semantic_parse"),
    "cstop_artificial": ("utterance", "semantic_parse"),
}


def fix_header_factory(dataset_name=None, uniform=False, locale=None):
    def fix_headers(examples):
        new_examples = {}
        if uniform:
            columns = parsing_name_mapping[dataset_name]
            mapping = {columns[0]: "utterance", columns[1]: "semantic_parse"}
            if len(columns) == 3:
                mapping[columns[2]] = "locale"
            elif locale:
                new_examples["locale"] = locale
        for header in examples.keys():
            f_header = header.strip()
            if not uniform:
                new_examples[f_header] = examples[header]
            elif f_header in mapping:
                new_examples[mapping[f_header]] = examples[header]
        return new_examples

    return fix_headers


langs = ["en", "de", "es", "fr", "hi", "th"]
splits = ["eval", "test", "train"]
split_dict = {}
for lang in langs:
    for split in splits:
        split_dict[f"{split}_{lang}"] = f"./mtop/{lang}/{split}.txt.csv"

d = load_dataset("csv", data_files=split_dict)
d.map(fix_header_factory()).push_to_hub("WillHeld/mtop")
uniform_mtop = d.map(
    fix_header_factory(dataset_name="mtop", uniform=True),
    remove_columns=[
        column
        for column in d["train_en"].column_names
        if column not in ["utterance", "semantic_parse", "locale"]
    ],
)

splits = ["validation", "test", "train"]
split_dict = {}
for split in splits:
    split_dict[f"{split}"] = f"./cstop/{split}.tsv.csv"

d = load_dataset("csv", data_files=split_dict)
d.map(fix_header_factory()).push_to_hub("WillHeld/hinglish_top")
uniform_hinglish_top = d.map(
    fix_header_factory(dataset_name="hinglish_top", uniform=True, locale="hin_en"),
    remove_columns=[
        column
        for column in d["train"].column_names
        if column not in ["utterance", "semantic_parse", "locale"]
    ],
)

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
d.map(fix_header_factory()).push_to_hub("WillHeld/top_v2")
uniform_top_v2 = d.map(
    fix_header_factory(dataset_name="top_v2", uniform=True, locale="en"),
    remove_columns=[
        column
        for column in d["train"].column_names
        if column not in ["utterance", "semantic_parse", "locale"]
    ],
)

splits = ["eval", "test", "train"]
split_dict = {}
for split in splits:
    split_dict[split] = f"./CSTOP/CSTOP_{split}.tsv.csv"
d = load_dataset("csv", data_files=split_dict)
d.map(fix_header_factory()).push_to_hub("WillHeld/cstop")
uniform_cstop = d.map(
    fix_header_factory(dataset_name="cstop", uniform=True, locale="spa_en"),
    remove_columns=[
        column
        for column in d["train"].column_names
        if column not in ["utterance", "semantic_parse", "locale"]
    ],
)
print(uniform_cstop)

splits = ["eval", "test", "train"]
split_dict = {}
for split in splits:
    split_dict[split] = f"./cstop_artificial/{split}.csv"
d = load_dataset("csv", data_files=split_dict)
d.map(fix_header_factory()).push_to_hub("WillHeld/cstop_artificial")
uniform_cstop_a = d.map(
    fix_header_factory(dataset_name="cstop_artificial", uniform=True, locale="en"),
    remove_columns=[
        column
        for column in d["train"].column_names
        if column not in ["utterance", "semantic_parse", "locale"]
    ],
)

uniform_dataset = uniform_mtop
for split in uniform_cstop:
    uniform_dataset[split + "_cstop"] = uniform_cstop[split]

for split in uniform_top_v2:
    uniform_dataset[split + "_top_v2"] = uniform_top_v2[split]

for split in uniform_hinglish_top:
    uniform_dataset[split + "_hinglish_top"] = uniform_hinglish_top[split]

for split in uniform_cstop_a:
    uniform_dataset[split + "_cstop_artificial"] = uniform_cstop_a[split]

uniform_dataset.push_to_hub("WillHeld/uniform_top")
