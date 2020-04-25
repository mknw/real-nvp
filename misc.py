
import numpy as np
from PIL import Image


def sv_image(x, name='sample.png'):
    x = np.asarray(x)
    # import ipdb; ipdb.set_trace()
    # x = np.reshape(x, (x.shape[1], x.shape[2]))
    x = x[0]
    # x = np.concatenate([x, x, x], axis=-1)

    # img = (x * 255).astype(np.uint8)

    img = Image.fromarray(x, mode='L')
    img.save(name)

    return img
