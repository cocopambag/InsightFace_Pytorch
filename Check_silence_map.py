from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import cv2
import numpy as np

import os
import random
import argparse
import shutil
import pandas as pd
from Face_Detector import Detector
from config import get_config
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank
from pathlib import Path
from PIL import Image
import torch
from matplotlib import pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', dest='test_path', type=str)
    parser.add_argument('-s', dest='store_path', type=str)
    parser.add_argument('-th', '--threshold', help='threshold to decide identical faces', default=1.54, type=float)
    parser.add_argument("-u", "--update", help="whether perform update the facebank", action="store_true",
                        default=False)
    parser.add_argument("-tta", "--tta", help="whether test time augmentation", action="store_true", default=False)
    parser.add_argument("-c", "--score", help="whether show the confidence score", action="store_true")
    parser.add_argument("-m", "--mobile", help="Use MobileFaceNet", default=False)

    args = parser.parse_args()
    test_path = args.test_path
    store_path = args.store_path

    if not os.path.exists(store_path):
        print("Don't exist test path:", store_path)
        exit()
    if not os.path.exists(test_path):
        print("Don't exist save path:", test_path)
        exit()

    D = Detector(thresh=0.6)
    print('Retinaface loaded')

    conf = get_config(False)

    conf['facebank_path'] = Path(test_path)

    conf.use_mobilfacenet = args.mobile

    learner = face_learner(conf, True)
    learner.threshold = args.threshold
    if conf.device.type == 'cpu':
        learner.load_state(conf, 'cpu_final.pth', True, True)
    elif conf.use_mobilfacenet:
        learner.load_state(conf, 'mobile.pth', True, True)
    else:
        learner.load_state(conf, 'final.pth', True, True)
    learner.model.eval()
    print('learner loaded')

    if not os.path.exists(test_path + '/' + 'facebank.pth'):
        targets, names = prepare_facebank(conf, learner.model, D, tta=args.tta)
        print('Make facebank')

    else:
        targets, names = load_facebank(conf)
        print('Load facebank')

    dictionaries = next(os.walk(test_path))[1]

    percentages = [0, ]
    average_correct = 0

    f = open("/media/user/새 볼륨/lfw_mobilenet/' + 'check.txt", 'w')

    for idx, dic in enumerate(dictionaries):  # Each people
        dic_path = test_path + '/' + dic

        images = next(os.walk(dic_path))[2]
        os.mkdir('/media/user/새 볼륨/lfw_mobilenet/' + dic)

        percentage = len(images)

        for image in images:

            image_path = dic_path + '/' + image

            img = cv2.imread(image_path)

            list_bbox, faces = D.detect(img)

            if len(list_bbox) == 0: # fail to search face
                print("remove", image_path)
                os.remove(image_path)
                continue

            list_bbox, faces = [list_bbox[0]], [faces[0]]

            r, score, imga, embs = learner.infer(conf, faces, targets, args.tta)

            emb = embs[0]
            embed = torch.sum(abs(emb), 1)

            embed.backward()

            saliency_map, _ = torch.max(imga.grad.data.abs(), dim=1)

            img = cv2.resize(cv2.cvtColor(np.array(faces[0]), cv2.COLOR_BGR2RGB), (112, 112), interpolation=cv2.INTER_CUBIC)

            # print("directory: {}\tclassify: {}\tscore: {:0.3f}".format(dic, names[r+1], score.item()))

            fig = plt.figure()
            ax1 = fig.add_subplot(221)
            ax2 = fig.add_subplot(222)
            ax1.imshow(img)
            ax2.imshow(saliency_map[0].cpu(), cmap=plt.cm.hot)
            # plt.show()

            c = 'o'
            if names[r+1] != dic:
                c = 'x' + names[r+1] + '0'
                percentage -= 1
                #print(image_path)
                f.write(image_path + '\n')
            plt.savefig('/media/user/새 볼륨/lfw_mobilenet/' + dic + '/' + c + image + '.jpg')
            plt.close(fig)


        # print(dic, percentage/float(len(images)))
        percentages.append(percentage/float(len(images)))
        average_correct += percentage/float(len(images))
        if idx % 1000 == 0:
            print(idx, average_correct/(idx+1))

    print("final:", average_correct/(idx+1))
    df = pd.DataFrame(percentages, index=names)
    df.to_csv('/media/user/새 볼륨/lfw_mobilenet/' + 'check.csv')
    f.close()

    del D
    del learner