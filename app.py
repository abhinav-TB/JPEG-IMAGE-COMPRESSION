#!/usr/bin/env python3
import numpy as np
import cv2
import scipy.fftpack as fftpack
import zlib
from utils import *

class jpeg_encoder:
    
    def __init__(self, im,quants):
        self.image = im
        self.quants = quants
        super().__init__()

    def encode_quant(self,quant):

        return (self.enc / quant).astype(np.int)

    def decode_quant(self,quant):
        return (self.encq * quant).astype(float)

    def encode_dct(self,bx, by):
        new_shape = (
            self.image.shape[0] // bx * bx,
            self.image.shape[1] // by * by,
            3
        )
        new = self.image[
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


    def decode_dct(self, bx, by):
        return fftpack.idctn(self.decq, axes=[1, 3], norm='ortho'
                            ).reshape((
                                self.decq.shape[0]*bx,
                                self.decq.shape[2]*by,
                                3
                            ))
    
    def encode_zip(self):
        return zlib.compress(self.encq.astype(np.int8).tobytes())


    def decode_zip(self):
        return np.frombuffer(zlib.decompress(self.encz), dtype=np.int8).astype(float).reshape(self.encq.shape)

    
if __name__ == "__main__":
    im = cv2.imread("IMG_0108.JPG")
    Ycr = rgb2ycbcr(im);
    obj=jpeg_encoder(Ycr,[5])
    quants = [5]  # [0.5, 1, 2, 5, 10]
    blocks = [(8, 8)]  # [(2, 8), (8, 8), (16, 16), (32, 32), (200, 200)]
    for qscale in quants:
        for bx, by in blocks:

            quant = (
                (np.ones((bx, by)) * (qscale * qscale))
                .clip(-100, 100)  # to prevent clipping
                .reshape((1, bx, 1, by, 1))
            )
            obj.enc = obj.encode_dct(bx, by)
            obj.encq = obj.encode_quant(quant)
            obj.encz = obj.encode_zip()
            obj.decz = obj.decode_zip()
            obj.decq = obj.decode_quant(quant)
            obj.dec = obj.decode_dct(bx, by)
            img_bgr = ycbcr2rgb(obj.dec)
            cv2.imwrite("IMG_0108_recompressed_quant_{}_block_{}x{}.jpeg".format(
                qscale, bx, by), img_bgr.astype(np.uint8))

