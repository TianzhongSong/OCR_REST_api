# -*- coding:utf-8 -*-
import cv2
import numpy as np
import heapq


def sort_box(box):
    """
    对box排序,及页面进行排版
    text_recs[index, 0] = x1
    text_recs[index, 1] = y1
    text_recs[index, 2] = x2
    text_recs[index, 3] = y2
    text_recs[index, 4] = x3
    text_recs[index, 5] = y3
    text_recs[index, 6] = x4
    text_recs[index, 7] = y4
    """

    box = sorted(box, key=lambda x: sum([x[1], x[3], x[5], x[7]]))
    return box


def correct_box(boxes, im, process=False):
    outBoxes = []
    img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    for box in boxes:
        box[0] = box[0] if box[0] >= box[4] else box[4]  # x1
        box[1] = box[1] if box[1] >= box[3] else box[3]  # y1
        box[2] = box[2] if box[2] >= box[6] else box[6]  # x2
        box[5] = box[5] if box[5] >= box[7] else box[7]  # y3
        if box[0] >= 5:
            box[0] -= 5
        else:
            box[0] = 0
        if box[1] >= 5:
            box[1] -= 5
        else:
            box[1] = 0
        if im.shape[1] - box[2] >= 5:
            box[2] += 5
        else:
            box[2] = im.shape[1]

        if im.shape[0] - box[5] >= 5:
            box[5] += 5
        else:
            box[5] = im.shape[0]

        box[3] = box[1]
        box[4] = box[0]
        box[6] = box[2]
        box[7] = box[5]
        outBoxes.append(box)

        if process:
            tmp = img[box[1]:box[7], box[0]:box[6]]
            hist = cv2.calcHist([tmp], [0], None, [256], [0, 256])
            hists = []
            for i in range(256):
                hists.append(int(hist[i]))
            tenLargest = heapq.nlargest(10, hists)
            # print(tenLargest)
            largest = tenLargest[0]
            largetsIndex = hists.index(largest)
            # print(largetsIndex)
            secodLargest = 0
            index = 1
            while secodLargest == 0 and index <= 9:
                if abs(largetsIndex - hists.index(tenLargest[index])) > 10:
                    secodLargest = tenLargest[index]
                    if tenLargest[index] == 0:
                        secodLargest = 1
                index += 1
            nb1Index = hists.index(largest)
            nb2Index = hists.index(secodLargest)
            thresh = (nb1Index + nb2Index) // 2
            th2 = 0
            # print(nb1Index, nb2Index)
            if nb1Index > nb2Index:
                th = thresh if thresh < nb1Index - th2 else nb1Index - th2

            else:
                tmp = 255 - tmp
                t = 255 - thresh
                th = t if t < 255 - nb1Index - th2 else t - th2
            ret, tmp = cv2.threshold(tmp, th, 255, cv2.THRESH_BINARY)
            mask = np.zeros((tmp.shape[0], tmp.shape[1], 3), dtype='uint8')
            for i in range(3):
                mask[:, :, i] = tmp
            im[box[1]:box[7], box[0]:box[6], :] = mask
    return outBoxes, im
