"""Physical window geometry shared by preprocessing, model, and evaluation."""


def window_starts(width, window_width, stride, *, shift_last=False):
    if width < window_width or min(window_width, stride) < 1:
        raise ValueError('Image width must cover a positive window; stride must be positive')
    starts = list(range(0, width - window_width + 1, stride))
    if shift_last and starts[-1] != width - window_width:
        starts.append(width - window_width)
    return starts
