import numpy as np

# ITU-R BT.601 (same as mpeg-pcc-dmetric tool)
bt601_rgb_to_yuv_m = np.array([[.299, .587, .114],
                           [-.147, -.289, .436],
                           [.615, -.515, -.100]]).T
bt601_yuv_to_rgb_m = np.linalg.inv(bt601_rgb_to_yuv_m)
# ITU-R BT.709 (same as mpeg-pcc-dmetric tool)
bt709_rgb_to_yuv_m = np.array([[.2126, .7152, .0722],
                           [-.1146, -.3854, .5000],
                           [.5000, -.4542, -.0458]]).T
bt709_yuv_to_rgb_m = np.linalg.inv(bt709_rgb_to_yuv_m)

rgb_to_yuv_m = {
    'bt601': bt601_rgb_to_yuv_m,
    'bt709': bt709_rgb_to_yuv_m,
}
yuv_to_rgb_m = {
    'bt601': bt601_yuv_to_rgb_m,
    'bt709': bt709_yuv_to_rgb_m,
}


def check_mode(mode):
    assert mode in ['bt601', 'bt709']


# arr : [n, 3]
# output : [n, 3]
def rgb_to_yuv(rgb, mode='bt709'):
    yuv = rgb @ rgb_to_yuv_m[mode]
    yuv[:, 1:] += 128
    return np.clip(yuv, 0.0, 255.0)


