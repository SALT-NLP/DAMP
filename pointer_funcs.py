from itertools import zip_longest


def pointer_process(source_in, target_in, vocabulary):
    """Replaces out-of-vocabulary words in source and target text with <unk-N>,
    where N in is the position of the word in the source sequence.
    """

    def format_pt(pos):
        return "<pt-{}>".format(pos)

    outs = []
    for seq_num, (source_seq, target_seq) in enumerate(
        zip_longest(source_in, target_in)
    ):
        target_seq_out = []

        word_to_pos = dict()
        for position, token in enumerate(source_seq.strip().split()):
            if token in word_to_pos:
                word_to_pos[token].append(position)
            else:
                word_to_pos[token] = [position]

        for token in target_seq.strip().split():
            if token in word_to_pos:
                token_out = format_unk(word_to_pos[token].pop(0))
            else:
                token_out = token
            target_seq_out.append(token_out)
        outs.append(target_seq_out)
    return outs
