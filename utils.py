import numpy as np
import cv2
from collections import OrderedDict 
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

def runLengthEncoding(message):
    encoded_message = []
    i = 0
   
    while (i <= len(message)-1): 
        count = 1
        ch = message[i] 
        j = i 
        while (j < len(message)-1): 
            if (message[j] == message[j+1]): 
                count = count+1
                j = j+1
            else: 
                break
        encoded_message.append(ch)
        encoded_message.append(count)
        i = j+1
    return encoded_message 

def runLengthDecoding(input):
    ans=[]
    for i in (0,len(input)-2,2):
        
        for j in (1,input[i+1]):
            ans.append(input[i])
    return ans


    
    