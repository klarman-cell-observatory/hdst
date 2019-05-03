from functools import partial

import itertools as it

import logging

import os

from imageio import imread, imwrite

import numpy as np

import pandas as pd

from scipy.ndimage.interpolation import zoom
from scipy.signal import fftconvolve


__all__ = ['align']

LOG = logging.getLogger(__package__)

SOBELX = np.matrix([[-1, 0, 1]]).T * np.matrix([[1, 2, 1]])
SOBELY = SOBELX.T


def sobel(im, amplify):
    """Applies a sobel filter to the input image.
    """
    if len(im.shape) == 2:
        fx, fy = SOBELX, SOBELY
        pad_width = (1,)
    elif len(im.shape) == 3:
        fx, fy = [np.stack([np.array(f)], axis=2) for f in (SOBELX, SOBELY)]
        pad_width = ((1, 1), (1, 1), (0, 0))
    else:
        raise ValueError('Invalid image dimensions')

    gx = fftconvolve(im, fx, mode='valid')
    gy = fftconvolve(im, fy, mode='valid')

    return np.pad(
        (gx * gx + gy * gy) ** (1 / (1 + np.exp(amplify))),
        pad_width,
        mode='edge',
    )


def zoomto(shape, im):
    """Zoom image to a given shape.
    """
    dims = len(im.shape)
    if dims not in [2, 3]:
        raise ValueError('Invalid image dimensions')

    zoom_factor = min([
        target / current
        for (target, current) in zip(shape, im.shape) if target != -1
    ])

    if zoom_factor == 1.:
        return im, 1.

    zoom_seq = [1] * dims
    zoom_seq[:2] = [zoom_factor] * 2
    return zoom(im, zoom=zoom_seq, order=1, mode='nearest'), zoom_factor


def ma(seq, length):
    """Computes the moving average of a 1-d sequence.
    """
    return np.pad(
        np.convolve(seq, [1 / length] * length, mode='valid'),
        (length // 2,),
        mode='edge',
    )


def deriv(seq):
    """Computes the derivative of a 1-d sequence using finite differences
    approximation.
    """
    return np.pad(
        np.convolve(seq, [1, 0, -1], mode='valid'),
        (1,),
        mode='edge',
    )


def getbounds(im, ymax=None, xmax=None):
    """Estimates the top and left boundaries of a light, rectangular object in a
    flattened image by optimizing the derivatives of the col and row sums of
    the image's intensity function.
    """
    ymax, xmax = [
        a if a is not None else s
        for (a, s) in zip((ymax, xmax), im.shape)
    ]
    y, x = (np.sum(im, axis=i) for i in (1, 0))
    dy, dx = (ma(deriv(a), max(1, len(a) // 100)) for a in (y, x))
    top, left = [np.argmax(d[:m]) for (d, m) in zip((dy, dx), (ymax, xmax))]
    return (top, left), (dy, dx)


def restriction(H, W, x1s, t):
    """Given x1s and t, computes the other xss:s (s.t. them forming a rectangle
    with height H and width W).
    """
    x11, x12 = x1s
    x2s = [x11 + W * np.sin(t), x12 + W * np.cos(t)]
    x3s = [x11 + H * np.cos(t), x12 - H * np.sin(t)]
    x4s = [x2s[0] + x3s[0] - x11, x2s[1] + x3s[1] - x12]
    return [x1s, x2s, x3s, x4s]


def drestricted_cost(H, W, yss, x1s, t):
    """Derivative of restricted cost function w.r.t. to x1s and t.
    """
    dx1s, dx2s, dx3s, dx4s = dcost(yss, restriction(H, W, x1s, t))
    dx1sr = [
        dx1s[0] + dx2s[0] + dx3s[0] + dx4s[0],
        dx1s[1] + dx2s[1] + dx3s[1] + dx4s[1],
    ]
    dt = dx2s[0] * W * np.cos(t) - dx2s[1] * W * np.sin(t) + \
        -dx3s[0] * H * np.sin(t) - dx3s[1] * H * np.cos(t) + \
        dx4s[0] * (W * np.cos(t) - H * np.sin(t)) + \
        dx4s[1] * (-W * np.sin(t) - H * np.cos(t))
    return [dx1sr, dt]


def restricted_cost(H, W, yss, x1s, t):
    """Restricted cost function.
    """
    return cost(yss, restriction(H, W, x1s, t))


def dcost(yss, xss):
    """Derivative of the cost function w.r.t. to the xss.
    """
    return [
        [2 * (x - y) for (x, y) in zip(xs, ys)]
        for (xs, ys) in zip(xss, yss)
    ]


def cost(yss, xss):
    """Euclidean cost function.
    """
    return sum(
        sum((x - y) ** 2 for (x, y) in zip(xs, ys))
        for (xs, ys) in zip(xss, yss)
    )


def optimize_cost(H, W, yss, x1s0, t0):
    """Optimize cost by gradient descent.
    """
    x1s = x1s0
    t = t0
    f = partial(restricted_cost, H, W, yss)
    df = partial(drestricted_cost, H, W, yss)
    for i in range(10000):
        dx1s, dt = df(x1s, t)
        x1s[0] -= 1e-3 * dx1s[0]
        x1s[1] -= 1e-3 * dx1s[1]
        t -= 1e-9 * dt
        LOG.debug('Iteration %d, loss=%.2e', i, f(x1s, t))
        LOG.debug('-----------------------')
        LOG.debug('x1s: %s', x1s)
        LOG.debug('dx1s: %s', dx1s)
        LOG.debug('t: %.3f', t)
        LOG.debug('dt: %.3f', dt)
    return x1s, t


def align(
        im_file,
        im_size,
        win_size,
        annotate=False,
        output_directory=None,
):
    """Runs the frame detection.
    """
    if output_directory is None:
        output_directory = '.'
    elif not os.path.exists(output_directory):
        os.makedirs(output_directory)

    im_file_no_ext = os.path.basename(im_file)[:-(im_file[::-1].find('.') + 1)]

    def _go(im_):
        (t1, l1), (dy1, dx1) = getbounds(im_, *[s // 8 for s in im_.shape])
        slices = [
            slice(max(c - s // 2, 0), c + (s + 1) // 2 + 1)
            for (c, s) in zip((t1, l1), win_size)
        ]
        win = im_[tuple(slices)]
        (t2, l2), (dy2, dx2) = getbounds(win)
        return [slices[0].start + t2, slices[1].start + l2]

    def _annotate(spots, image):
        maxval = np.iinfo(image.dtype).max
        image[spots[0, :], spots[1, :]] = (
            [maxval, 0, 0]
            if image.shape[-1] == 3 else
            [maxval, 0, 0, maxval]
            if image.shape[-1] == 4 else
            maxval
        )
        save_path = os.path.join(
            output_directory,
            f'{im_file_no_ext}.annotated.tif',
        )
        LOG.info('Saving annotated image to %s', save_path)
        imwrite(save_path, image)

    im = imread(im_file)

    if annotate:
        _annotate = partial(_annotate, image=im.copy())
    else:
        _annotate = lambda *_: None

    LOG.info('Scaling image to %dx%d', *im_size)
    im, zoom_factor = zoomto(im_size, im)

    LOG.info('Applying sobel filter and flattening')
    im = np.sum(sobel(im, 1.5), axis=2)

    LOG.info('Running bounds detection')

    LOG.debug('Running bounds detection on the top-left corner')
    tl = _go(im)

    LOG.debug('Running bounds detection on the top-right corner')
    tr = _go(im[:, ::-1])
    tr[1] = im.shape[1] - tr[1] - 1

    LOG.debug('Running bounds detection on the bottom-left corner')
    bl = _go(im[::-1, :])
    bl[0] = im.shape[0] - bl[0] - 1

    LOG.debug('Running bounds detection on the bottom-right corner')
    br = _go(im[::-1, ::-1])
    br[0] = im.shape[0] - br[0] - 1
    br[1] = im.shape[1] - br[1] - 1

    tl, tr, bl, br = [[a / zoom_factor for a in b] for b in (tl, tr, bl, br)]
    LOG.info('Unaligned result: top-left=%s', tl)
    LOG.info('Unaligned result: top-right=%s', tr)
    LOG.info('Unaligned result: bottom-left=%s', bl)
    LOG.info('Unaligned result: bottom-right=%s', br)

    array_size_px = [
        (
            np.sqrt((tl[1] - bl[1]) ** 2 + (tl[0] - bl[0]) ** 2) +
            np.sqrt((tr[1] - br[1]) ** 2 + (tr[0] - br[0]) ** 2)
        ) / 2,
        (
            np.sqrt((tr[1] - tl[1]) ** 2 + (tr[0] - tr[0]) ** 2) +
            np.sqrt((br[1] - bl[1]) ** 2 + (br[0] - br[0]) ** 2)
        ) / 2,
    ]
    x1s, t = optimize_cost(
        *array_size_px,
        [tl, tr, bl, br],
        tl,
        0,
    )
    tl_, tr_, bl_, br_ = restriction(*array_size_px, x1s, t)

    print('Rotation=%.3f rad' % t)
    print('Top-left=%s' % tl_)
    print('Top-right=%s' % tr_)
    print('Bottom-left=%s' % bl_)
    print('Bottom-right=%s' % br_)

    spots = np.concatenate(
        list(map(
            np.transpose,
            map(np.matrix, it.product(range(783), range(1918), [1]))
        )),
        axis=1,
    ).astype(np.float64)

    spot_labels = spots[:2, :].copy()

    spots[1, :] += 0.5 * (0.5 + spots[0, :] % 2)
    spots[0, :] += 0.5
    spots[1, :] *= array_size_px[1] / 1918
    spots[0, :] *= array_size_px[0] / 783

    R = np.matrix([
        [np.cos(t), np.sin(t), 0],
        [-np.sin(t), np.cos(t), 0],
        [0, 0, 1],
    ])
    T = np.matrix([
        [1, 0, tl_[0]],
        [0, 1, tl_[1]],
        [0, 0, 1],
    ])

    spots = (T * R * spots)
    spots = np.round(spots).astype(int)

    # index labels from 1
    spot_labels += 1

    df = pd.concat(
        [
            pd.DataFrame(
                spot_labels.T,
                columns=['spot_y', 'spot_x'],
                dtype=int,
            ),
            pd.DataFrame(
                spots[:2, :].T,
                columns=['spot_px_y', 'spot_px_x'],
                dtype=int,
            ),
        ],
        axis=1,
    )

    save_path = os.path.join(output_directory, f'{im_file_no_ext}.tsv')
    LOG.info('Saving spots file to %s', save_path)
    df.to_csv(save_path, index=None, sep='\t')

    _annotate(spots)
