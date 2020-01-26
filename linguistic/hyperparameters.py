class HyperParameters:
    speakers = list(set(['p'+str(i) for i in range(225,242)])-set(['p235']))
    num_speakers = len(speakers)
    dropout = 0.5
    preemphasis = 0.97
    data = "LJSpeech-1.1/"
    sr = 22050
    frame_shift = 0.0125
    frame_len = 0.05
    hop_len = int(sr * frame_shift)
    window_len = int(sr * frame_len)
    n_fft = 2048
    r = 2
    n_mels = 80
    vocab = "PE abcdefghijklmnopqrstuvwxyz'.?"
    k = 16
    momentum = 0.99
    eps = 1e-3