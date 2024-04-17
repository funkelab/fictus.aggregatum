import numpy as np


def wav2RGB(wavelength):
    """
    Code adapted from: [here](http://codingmess.blogspot.com/2009/05/conversion-of-wavelength-in-nanometers.html)
    See [here](https://www.fourmilab.ch/documents/specrend/) for more information.
    """
    w = int(wavelength)

    # colour
    if w >= 380 and w < 440:
        R = -(w - 440.0) / (440.0 - 350.0)
        G = 0.0
        B = 1.0
    elif w >= 440 and w < 490:
        R = 0.0
        G = (w - 440.0) / (490.0 - 440.0)
        B = 1.0
    elif w >= 490 and w < 510:
        R = 0.0
        G = 1.0
        B = -(w - 510.0) / (510.0 - 490.0)
    elif w >= 510 and w < 580:
        R = (w - 510.0) / (580.0 - 510.0)
        G = 1.0
        B = 0.0
    elif w >= 580 and w < 645:
        R = 1.0
        G = -(w - 645.0) / (645.0 - 580.0)
        B = 0.0
    elif w >= 645 and w <= 780:
        R = 1.0
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0

    # intensity correction
    # if w >= 380 and w < 420:
    #     SSS = 0.3 + 0.7*(w - 350) / (420 - 350)
    # elif w >= 420 and w <= 700:
    #     SSS = 1.0
    # elif w > 700 and w <= 780:
    #     SSS = 0.3 + 0.7*(780 - w) / (780 - 700)
    # else:
    #     SSS = 0.0
    # # SSS *= 255

    return R, G, B  # [SSS*R, SSS*G, SSS*B]


def apply_spectra(image, wavelength):
    rgb = wav2RGB(wavelength)  # returns r,g,b values between 0 and 1

    image = np.stack([image * c for c in rgb], axis=-1)

    return image
