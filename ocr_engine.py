import argparse
import os
import string

import cv2
import torch

from CRAFT_pytorch import file_utils
from CRAFT_pytorch.craft import CRAFT
from CRAFT_pytorch.test import copyStateDict, test_net
import torch.backends.cudnn as cudnn

from deep_text_recognition_benchmark.dataset import AlignCollate, SuperRawDataset
from deep_text_recognition_benchmark.model import Model
from deep_text_recognition_benchmark.utils import CTCLabelConverter, AttnLabelConverter

import torch.nn.functional as F
import time

import pandas as pd
from pathlib import Path

def get_list_of_corners_cor(bbox):
    """

    :param bbox:
    :return:
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = [int(tmp) for tmp in bbox.flatten()]
    return x1, y1, x2, y2, x3, y3, x4, y4


def get_corners(bbox):
    """
    Return the corners coordinates of bbox
    :param bbox:
    :return:
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = get_list_of_corners_cor(bbox)

    top_left_x = min([x1,x2,x3,x4])
    top_left_y = min([y1,y2,y3,y4])
    bot_right_x = max([x1,x2,x3,x4])
    bot_right_y = max([y1,y2,y3,y4])

    return top_left_x, top_left_y, bot_right_x, bot_right_y




def crop_bbox(image, bbox):
    """
    Crop a bounding box
    :param image:
    :param bbox:
    :return:
    """
    top_left_x, top_left_y, bot_right_x, bot_right_y = get_corners(bbox)

    return image[top_left_y:bot_right_y+1, top_left_x:bot_right_x+1]

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark_all_eval', action='store_true', help='evaluate 10 benchmark evaluation datasets')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--new_characters', type=str, default='', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    parser.add_argument('--baiduCTC', action='store_true', help='for data_filtering_off mode')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, default="TPS", help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default="ResNet", help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, default="BiLSTM", help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default="Attn", help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    return parser.parse_known_args()[0]



# Slightly modified version of original script

class DeepText:
    def __init__(self, lang):
        
        self.lang = lang
        self.opt = parse_arguments()
        self.THAI_RECOGNIZER_PATH = "deep_text_recognition_benchmark/saved_models/Thai/best_norm_ED.pth"
        self.ENG_RECOGNIZER_PATH = "deep_text_recognition_benchmark/saved_models/English/best_accuracy.pth"
        self.ENG_PUBLIC_RECOGNIZER_PATH = "deep_text_recognition_benchmark/saved_models/Public/TPS-ResNet-BiLSTM-Attn-case-sensitive.pth"
    
        """ vocab / character number configuration """

        if True:
            self.opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).
        
        if not self.lang == "english_public":
            self.opt.character += "กขคฆงจฉชฌญฎฐฒณดตถทธนบปผพฟภมยรลวศษสหฬอฮ"

        cudnn.benchmark = True
        cudnn.deterministic = True
        self.opt.num_gpu = torch.cuda.device_count()

        """ model configuration """
        if 'CTC' in self.opt.Prediction:
            self.converter = CTCLabelConverter(self.opt.character)
        else:
            self.converter = AttnLabelConverter(self.opt.character)
        self.opt.num_class = len(self.converter.character)

        if self.opt.rgb:
            self.opt.input_channel = 3
            
        self.recognizer = Model(self.opt)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.recognizer = torch.nn.DataParallel(self.recognizer).to(self.device)

        # load model
        
        if self.lang == "thai":
            if not Path(self.THAI_RECOGNIZER_PATH).is_file():
                self.lang = "english_public"
                print("Couldn't find the model :", self.THAI_RECOGNIZER_PATH)
            else:
                print('loading pretrained model from %s' % self.THAI_RECOGNIZER_PATH)
                self.recognizer.load_state_dict(torch.load(self.THAI_RECOGNIZER_PATH, map_location=self.device))
                
        if self.lang == "english":
            if not Path(self.ENG_RECOGNIZER_PATH).is_file():
                self.lang = "english_public"
                print("Couldn't find the model :", self.ENG_RECOGNIZER_PATH)
            else:
                print('loading pretrained model from %s' % self.ENG_RECOGNIZER_PATH)
                self.recognizer.load_state_dict(torch.load(self.ENG_RECOGNIZER_PATH, map_location=self.device))
                
        if self.lang == "english_public":
            print('loading pretrained model from %s' % self.ENG_PUBLIC_RECOGNIZER_PATH)
            self.recognizer.load_state_dict(torch.load(self.ENG_PUBLIC_RECOGNIZER_PATH, map_location=self.device))
            
        self.recognizer.eval()
        self.alignCollate = AlignCollate(imgH=self.opt.imgH, imgW=self.opt.imgW, keep_ratio_with_pad=self.opt.PAD)
    
    
    def detect_text(self, cropped_bboxes, bboxes, text_only=True):
        
        result = self.__detect_text(cropped_bboxes, bboxes, bboxes)
        
        text = []
        if text_only:
            for i in result[1]:
                if i.get('text', False):
                    text.append(i['text'])
            
            return ''.join(text)
        else:
            return result
            
        
    def __detect_text(self, cropped_bboxes, bboxes, polys):
        
        dataset = SuperRawDataset(cropped_bboxes, bboxes, polys)

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.opt.batch_size,
            shuffle=False,
            num_workers=1,
            collate_fn=self.alignCollate, pin_memory=True)

        results = []

        with torch.no_grad():
            for bbox_tensors, bboxes_polys in data_loader:
                batch_size = bbox_tensors.size(0)
                bbox_tensors = bbox_tensors.to(self.device)
                # For max length prediction
                length_for_pred = torch.IntTensor([self.opt.batch_max_length] * batch_size).to(self.device)
                text_for_pred = torch.LongTensor(batch_size, self.opt.batch_max_length + 1).fill_(0).to(self.device)

                if 'CTC' in self.opt.Prediction:
                    preds = self.recognizer(bbox_tensors, text_for_pred)

                    # Select max probabilty (greedy decoding) then decode index to character
                    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                    _, preds_index = preds.max(2)
                    # preds_index = preds_index.view(-1)
                    preds_str = self.converter.decode(preds_index, preds_size)

                else:
                    preds = self.recognizer(bbox_tensors, text_for_pred, is_train=False)

                    # select max probabilty (greedy decoding) then decode index to character
                    _, preds_index = preds.max(2)
                    preds_str = self.converter.decode(preds_index, length_for_pred)

                preds_prob = F.softmax(preds, dim=2)
                preds_max_prob, _ = preds_prob.max(dim=2)
                for pred, pred_max_prob, (bbox, poly) in zip(preds_str, preds_max_prob, bboxes_polys):
                    if 'Attn' in self.opt.Prediction:
                        pred_EOS = pred.find('[s]')
                        pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                        pred_max_prob = pred_max_prob[:pred_EOS]

                    # calculate confidence score (= multiply of pred_max_prob)
                    confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                    results.append({
                        "text": pred,
                        "conf": round(confidence_score.item(), 4)
                    })

        return results
