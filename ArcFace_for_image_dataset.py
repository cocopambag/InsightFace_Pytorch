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


def make_gallery(origin, save_path, gallery_num):
    '''
    :param origin: Original people folder name
    :param save_path: Folder name to be saved
    :param gallery_num: The number of images to be included in the gallery
    :return: File name used for gallery
    '''
    try:
        name = origin.split('/')[-1]

    except:
        name = None

    if not name:
        print("Don't exist original file.")
        exit()
    else:
        try:
            os.mkdir(save_path + '/' + name)
        except:
            shutil.rmtree(save_path + '/' + name)
            os.mkdir(save_path + '/' + name)

    origin_f = next(os.walk(origin))[2]
    choice = [origin_f[a] for a in random.sample(range(0, len(origin_f)), gallery_num)]

    if save_path:
        for idx in choice:
            shutil.copyfile(origin + '/' + idx, save_path + '/' + name + '/' + idx)

    return choice


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', dest='test_path', type=str)
    parser.add_argument('-s', dest='save_path', type=str)
    parser.add_argument('-n', dest='gallery_num', type=int)
    parser.add_argument('-th', '--threshold', help='threshold to decide identical faces', default=1.54, type=float)
    parser.add_argument("-u", "--update", help="whether perform update the facebank", action="store_true",
                        default=False)
    parser.add_argument("-tta", "--tta", help="whether test time augmentation", action="store_true", default=False)
    parser.add_argument("-c", "--score", help="whether show the confidence score", action="store_true")
    parser.add_argument("-m", "--mobile", help="Use MobileFaceNet", default=False)

    args = parser.parse_args()
    test_path = args.test_path
    save_path = args.save_path
    gallery_num = args.gallery_num

    if not os.path.exists(test_path):
        print("Don't exist test path:", test_path)
        exit()
    if not os.path.exists(save_path):
        print("Don't exist save path:", save_path)
        os.mkdir(save_path)
        print("make a save_path: ", save_path)

    conf = get_config(False)

    conf['facebank_path'] = Path(save_path)

    dictionaries = next(os.walk(test_path))[1]
    names = dictionaries
    index = ['original', 'glasses', 'glasses_mask', 'mask']
    values = [[0, 0, 0, 0]]
    except_dic = dict()

    # choice images to be included in the gallery and make .npy file.
    for dic in dictionaries:  # Each people
        dic_path = test_path + '/' + dic
        save_folder_path = save_path + '/' + dic
        folders = next(os.walk(dic_path))[1]

        origin = dic_path + '/' + dic
        mask = dic_path + '/' + index[3]
        glasses = dic_path + '/' + index[1]
        m_s = dic_path + '/' + index[2]

        except_image = make_gallery(origin, save_path, gallery_num)
        except_dic[dic] = except_image

    D = Detector(thresh=0.9)
    print('Retinaface loaded')

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

    if args.update:
        targets, names = prepare_facebank(conf, learner.model, D, tta=args.tta)
        print('facebank updated')
    else:
        targets, names = load_facebank(conf)
        print('facebank loaded')

    for dic in dictionaries:  # Each people
        dic_path = test_path + '/' + dic
        folders = next(os.walk(dic_path))[1]
        result = [0, 0, 0, 0]
        current = None

        for folder in folders:
            folder_path = dic_path + '/' + folder
            except_file = []
            if folder == dic:
                except_file = except_dic[dic]

            if index[1] in folder and index[3] in folder:
                current = 2
            elif index[1] in folder:
                current = 1
            elif index[3] in folder:
                current = 3
            else:
                current = 0
            images = next(os.walk(folder_path))[2]
            num = len(images)
            for image in images:
                if image in except_file:
                    continue
                image_path = folder_path + '/' + image

                img = cv2.imread(image_path)

                list_bbox, faces = D.detect(img)
                # if len(list_bbox) > 1: # case of face images than 1
                #
                #     if
                #     faces = [faces[0]]
                # elif len(list_bbox) == 0: # fail to search face
                #     num -= 1
                #     print("remove", image_path)
                #     os.remove(image_path)
                #     continue
                if len(list_bbox) == 0: # fail to search face
                    num -= 1
                    print("remove", image_path)
                    os.remove(image_path)
                    continue

                r, score = learner.infer(conf, faces, targets, args.tta)

                try: # case of face images than 1
                    if names[r+1] != dic:
                        pass
                except:
                    if r[0] == r[1]:
                        r = r[0]
                    else:
                        continue

                if names[r+1] != dic:
                    # print(dic)
                    # print(names[r])
                    # cv2.imshow('fail', img)
                    # cv2.waitKey(10000)
                    pass
                else:
                    result[current] += 1

            result[current] /= float(num)

        print(dic, result)
        values.append(result)


    df = pd.DataFrame(values, index=names, columns=index)
    df.to_csv(save_path + '/' + 'check.csv')

    del D
    del learner
