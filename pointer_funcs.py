from itertools import zip_longest
import difflib


def get_closest_match(token, options, backup_options):
    additions = 10000
    best_replacement = None
    backward = False
    if len(options) > 0 and options[0].startswith(token):
        best_replacement = options[0]
    if best_replacement == None:
        for option in options:
            addition_distance = len(option.replace(token, "", 1))
            if token in option and addition_distance < additions:
                best_replacement = option
                additions = addition_distance
    if best_replacement == None:
        for option in backup_options:
            addition_distance = len(option.replace(token, "", 1))
            if token in option and addition_distance < additions:
                best_replacement = option
                additions = addition_distance
                backward = True
    return best_replacement, backward


def pointer_process(source_seq, target_seq):
    def format_pt(pos):
        return "<pt-{}>".format(pos)

    outs = []
    target_seq_out = []

    word_to_pos = dict()
    target_tokens = target_seq.strip().split()
    prev_offset = 0
    base_offset = 0

    for token in target_tokens:
        token = token.strip()
        if token.startswith("[") or token.startswith("]"):
            token_out = token
        else:
            matched_token, backward = get_closest_match(
                token,
                source_seq[prev_offset:].strip().split(),
                source_seq.strip().split(),
            )
            if backward:
                offset = source_seq.find(token)
            else:
                offset = prev_offset + source_seq[prev_offset:].find(token)

            source_seq = (
                source_seq[:offset]
                + " "
                + source_seq[offset : offset + len(token)]
                + " "
                + source_seq[offset + len(token) :]
            )
            if backward:
                print(source_seq, target_seq)
            base_offset += 2
            prev_offset = offset + len(token) + 2

    word_to_pos = dict()
    for position, token in enumerate(source_seq.strip().split()):
        token = token.strip()
        if token in word_to_pos:
            word_to_pos[token].append(position)
        else:
            word_to_pos[token] = [position]

    for token in target_tokens:
        token = token.strip()
        if token in word_to_pos:
            if token.startswith("[") or token.startswith("]"):
                print(source_seq, target_seq, token)
                sys.exit()
            token_out = format_pt(word_to_pos[token].pop(0))
        else:
            token_out = token
        target_seq_out.append(token_out)
    return " ".join(source_seq.strip().split()), " ".join(target_seq_out)
