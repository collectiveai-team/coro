from coro.core.realignment import realign_speaker_flicker


def assemble(words):
    return realign_speaker_flicker(list(words))
