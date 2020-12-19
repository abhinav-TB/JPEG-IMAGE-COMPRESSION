#!/usr/bin/env python3
import numpy as np
import cv2
import scipy.fftpack as fftpack
import zlib

class jpeg_encoder:
    
    def __init__(self, im,quants):
        self.image = im
        self.quants = quants
        super().__init__()

    def encode_quant(self,enc,quant):

        return (enc / quant).astype(np.int)

    def decode_quant(self,orig,quant):
        return (orig * quant).astype(float)

    def encode_dct(self,orig, bx, by):
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


    def decode_dct(self,orig, bx, by):
        return fftpack.idctn(orig, axes=[1, 3], norm='ortho'
                            ).reshape((
                                orig.shape[0]*bx,
                                orig.shape[2]*by,
                                3
                            ))
    
    def encode_zip(self,x):
        return zlib.compress(x.astype(np.int8).tobytes())


    def decode_zip(self,orig, shape):
        return np.frombuffer(zlib.decompress(orig), dtype=np.int8).astype(float).reshape(shape)


if __name__ == "__main__":
    im = cv2.imread("IMG_0108.JPG")
    obj=jpeg_encoder(im,[5])
    quants = [5]  # [0.5, 1, 2, 5, 10]
    blocks = [(8, 8)]  # [(2, 8), (8, 8), (16, 16), (32, 32), (200, 200)]
    for qscale in quants:
        for bx, by in blocks:

            quant = (
                (np.ones((bx, by)) * (qscale * qscale))
                .clip(-100, 100)  # to prevent clipping
                .reshape((1, bx, 1, by, 1))
            )
            enc = obj.encode_dct(im, bx, by)
            encq = obj.encode_quant(enc, quant)
            encz = obj.encode_zip(encq)
            decz = obj.decode_zip(encz, encq.shape)
            decq = obj.decode_quant(encq, quant)
            dec = obj.decode_dct(decq, bx, by)
            # dec = obj.ycbcr2rgb(dec)
            cv2.imwrite("IMG_0108_recompressed_quant_{}_block_{}x{}.jpeg".format(
                qscale, bx, by), dec.astype(np.uint8))

