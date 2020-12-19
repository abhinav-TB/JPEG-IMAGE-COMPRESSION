#!/usr/bin/env python3
import numpy as np
import cv2
import scipy.fftpack as fftpack
import zlib
def prev_16(x): return x >> 4 << 4


def encode_quant(orig, quant):
    # import code
    # code.interact(local=vars())
    return (orig / quant).astype(np.int)


def decode_quant(orig, quant):
    return (orig * quant).astype(float)


def encode_dct(orig, bx, by):
    new_shape = (
        orig.shape[0] // bx * bx,
        orig.shape[1] // by * by,
        3
    )
    new = orig[
        :new_shape[0],
        :new_shape[1]
    ].reshape((
        new_shape[0] // bx,
        bx,
        new_shape[1] // by,
        by,
        3
    ))
    return fftpack.dctn(new, axes=[1, 3], norm='ortho')


def decode_dct(orig, bx, by):
    return fftpack.idctn(orig, axes=[1, 3], norm='ortho'
                         ).reshape((
                             orig.shape[0]*bx,
                             orig.shape[2]*by,
                             3
                         ))


def encode_zip(x):
    return zlib.compress(x.astype(np.int8).tobytes())


def decode_zip(orig, shape):
    return np.frombuffer(zlib.decompress(orig), dtype=np.int8).astype(float).reshape(shape)


def rgb2ycbcr(im_rgb):
    im_rgb = im_rgb.astype(np.float32)
    im_ycrcb = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2YCR_CB)
    im_ycbcr = im_ycrcb[:, :, (0, 2, 1)].astype(np.float32)
    im_ycbcr[:, :, 0] = (im_ycbcr[:, :, 0]*(235-16)+16) / \
        255.0  # to [16/255, 235/255]
    im_ycbcr[:, :, 1:] = (im_ycbcr[:, :, 1:]*(240-16)+16) / \
        255.0  # to [16/255, 240/255]
    return im_ycbcr


def ycbcr2rgb(im_ycbcr):
    im_ycbcr = im_ycbcr.astype(np.float32)
    im_ycbcr[:, :, 0] = (im_ycbcr[:, :, 0]*255.0-16)/(235-16)  # to [0, 1]
    im_ycbcr[:, :, 1:] = (im_ycbcr[:, :, 1:]*255.0-16)/(240-16)  # to [0, 1]
    im_ycrcb = im_ycbcr[:, :, (0, 2, 1)].astype(np.float32)
    im_rgb = cv2.cvtColor(im_ycrcb, cv2.COLOR_YCR_CB2RGB)
    return im_rgb


if __name__ == '__main__':

    rgb = cv2.imread("IMG_0108.JPG")
    im = rgb2ycbcr(rgb)
    quants = [5]  # [0.5, 1, 2, 5, 10]
    blocks = [(8, 8)]  # [(2, 8), (8, 8), (16, 16), (32, 32), (200, 200)]
    for qscale in quants:
        for bx, by in blocks:

            quant = (
                (np.ones((bx, by)) * (qscale * qscale))
                .clip(-100, 100)  # to prevent clipping
                .reshape((1, bx, 1, by, 1))
            )
            enc = encode_dct(im, bx, by)
            encq = encode_quant(enc, quant)
            encz = encode_zip(encq)
            decz = decode_zip(encz, encq.shape)
            decq = decode_quant(encq, quant)
            dec = decode_dct(decq, bx, by)
            dec = ycbcr2rgb(dec)
            cv2.imwrite("IMG_0108_recompresse_quant_{}_block_{}x{}.jpeg".format(
                qscale, bx, by), dec.astype(np.uint8))

