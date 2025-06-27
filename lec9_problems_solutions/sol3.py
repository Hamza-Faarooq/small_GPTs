import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

# Recover merges
def recover_merges(mergeable_ranks):
    merges = [None] * len(mergeable_ranks)
    for token, rank in mergeable_ranks.items():
        if len(token) != 2: continue
        merges[rank] = tuple(token)
    return [m for m in merges if m is not None]

# Byte shuffle mapping
byte_shuffle = {i: enc._mergeable_ranks[bytes([i])] for i in range(256)}
