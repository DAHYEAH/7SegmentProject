import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from utils import CTCLabelConverter, AttnLabelConverter
from dataset import RawDataset, AlignCollate
from model import Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import cv2, os, json
import pandas as pd
import numpy as np

strings = []

def demo(opt):
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    model.eval()
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index, preds_size)

            else:
                preds = model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)


            log = open(f'./log_demo_result.txt', 'a')
            dashed_line = '-' * 80
            head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'
            
            print(f'{dashed_line}\n{head}\n{dashed_line}')
            log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                if 'Attn' in opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}')

                ##아래 네 줄은 커스텀한 내용##
                log.write(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}\n')
                if float(confidence_score) < 0.9 :  ## 오검출되는 부분 제거하기위해 
                    continue              ## confidence < 0.9 인 경우 제거 
                strings.append(img_name.split('.')[0] + '_' + pred)

            log.close()

def detecttext():
    # 옮겨서 저장하는 과정에서 resize 진행 
    test_img_path = "OCR/test_img/maternity/test.jpg"
    test_img = cv2.imread(test_img_path)
    test_img = cv2.resize(test_img,(1500,2000))
    test_img = np.where(test_img > 110, 255,0)
    cv2.imwrite('OCR/content/images/'+"test.jpg", test_img)

    # 글자 영역 넓게 지정 
    text_area_box = [
        [0, 700, 0, 250],  
        [0, 300, 0, 2000],
        [500, 750, 750, 1000 ],
        [1000, 1250, 750, 1000 ],
        [700, 1000, 1100, 1600],
        [0,1500,1700,2000]
    ]

    tmp_img = cv2.imread('OCR/content/images/'+"test.jpg")

    ############ 숫자 검출 ###############
    # 글자영역에 생기는 box 제거 
    # box안에 생기는 box 제거 (ex. 0의 경우)
    # 히스토그램 평활화, CLAHE, 
    image = tmp_img
    img_gray = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2GRAY)
    ret, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    for cnt in contours: 
        x, y, w, h = cv2.boundingRect(cnt) 
        if w*h < 1300 or w*h > 10000: 
            continue 

        is_inside = False
        for other_cnt in contours :
            ox, oy, ow, oh = cv2.boundingRect(other_cnt)
            # 현재 사각형이 다른 사각형에 포함되어 있는지 확인
            is_inside = ox < x < ox + ow and oy < y < oy + oh and ox < x + w < ox + ow and oy < y + h < oy + oh and (ow*oh>1800 and ow*oh<10000)
            if is_inside:
                break

        in_txt_area = False
        for txt_area in text_area_box :
            x1, x2, y1, y2 = txt_area
            # 글자영역에 있는지 확인 
            in_txt_area = x1 <= x <= x2 and y1 <= y <= y2 and x1 <= x + w <= x2 and y1 <= y + h <= y2 
            if in_txt_area :
                break
        
        if not is_inside and not in_txt_area :
            filtered_contours.append([x,y,w,h])

        img_cp = image[:]
        for cnt in filtered_contours:
            o_x, o_y, w, h = cnt 
            if w < 40 :
                if o_x > 30 :
                    x = o_x - 30 
                    w += 30
                else :
                    x = o_x
            else:
                if o_x > 10 :
                    x = o_x - 10 
                else : 
                    x = o_x 
            y = o_y - 10
            cv2.rectangle(img_cp, (x, y), (x+w+15, y+h+15), (0, 0, 255), 1) 
            cv2.imwrite(f'OCR/content/crop_img/{x+1}_{x+w+15}_{y+1}_{y+h+15}.jpg',image[y+1:y+h+15,x+1:x+w+15])

    # cv2.waitKey(0)

def makedict() :
    points = []
    int_points = []
    for i in strings :
        filename = i.split('/')[-1]
        # filename = filename.split('.')[0]
        points.append(filename.split('_'))
    
    for i in points :   
        int_points.append([int(k) for k in i])
    sorted_int_points = sorted(int_points, key=lambda x: (x[2], x[0]))

    # sorted(sorted_int_points[5:7])

    result = {}
    result['모돈번호'] = int(''.join([str(x[-1]) for x in sorted(sorted_int_points[0:5])]))

    result['출생일(년)'] = int(''.join([str(x[-1]) for x in sorted(sorted_int_points[5:11])][:2]))
    result['출생일(월)'] = int(''.join([str(x[-1]) for x in sorted(sorted_int_points[5:11])][2:4]))
    result['출생일(일)'] = int(''.join([str(x[-1]) for x in sorted(sorted_int_points[5:11])][4:6]))

    result['구입일(년)'] = int(''.join([str(x[-1]) for x in sorted(sorted_int_points[11:17])][:2]))
    result['구입일(월)'] = int(''.join([str(x[-1]) for x in sorted(sorted_int_points[11:17])][2:4]))
    result['구입일(일)'] = int(''.join([str(x[-1]) for x in sorted(sorted_int_points[11:17])][4:6]))

    result['분만예정일(월)'] = int(''.join([str(x[-1]) for x in sorted(sorted_int_points[17:21])][:2]))
    result['분만예정일(일)'] = int(''.join([str(x[-1]) for x in sorted(sorted_int_points[17:21])][2:4]))

    result['분만일(월)'] = int(''.join([str(x[-1]) for x in sorted(sorted_int_points[21:25])][:2]))
    result['분만일(일)'] = int(''.join([str(x[-1]) for x in sorted(sorted_int_points[21:25])][2:4]))

    result['총산자수'] = int(''.join([str(x[-1]) for x in sorted(sorted_int_points[25:31])][:2]))
    result['포유개시두수'] = int(''.join([str(x[-1]) for x in sorted(sorted_int_points[25:31])][2:4]))
    result['생시체중'] = float('.'.join([str(x[-1]) for x in sorted(sorted_int_points[25:31])][4:6]))

    result['이유일(월)'] = int(''.join([str(x[-1]) for x in sorted(sorted_int_points[31:35])][:2]))
    result['이유일(일)'] = int(''.join([str(x[-1]) for x in sorted(sorted_int_points[31:35])][2:4]))

    result['이유두수'] = int(''.join([str(x[-1]) for x in sorted(sorted_int_points[35:39])][:2]))
    result['이유체중'] = float('.'.join([str(x[-1]) for x in sorted(sorted_int_points[35:39])][2:4]))

    result['백신1(월)'] = int(''.join([str(x[-1]) for x in sorted(sorted_int_points[39:47])][:2]))
    result['백신1(일)'] = int(''.join([str(x[-1]) for x in sorted(sorted_int_points[39:47])][2:4]))
    result['백신2(월)'] = int(''.join([str(x[-1]) for x in sorted(sorted_int_points[39:47])][4:6]))
    result['백신2(일)'] = int(''.join([str(x[-1]) for x in sorted(sorted_int_points[39:47])][6:8]))

    result['백신3(월)'] = int(''.join([str(x[-1]) for x in sorted(sorted_int_points[47:55])][:2]))
    result['백신3(일)'] = int(''.join([str(x[-1]) for x in sorted(sorted_int_points[47:55])][2:4]))
    result['백신4(월)'] = int(''.join([str(x[-1]) for x in sorted(sorted_int_points[47:55])][4:6]))
    result['백신4(일)'] = int(''.join([str(x[-1]) for x in sorted(sorted_int_points[47:55])][6:8]))

    json_data = json.dumps(result, indent=1, ensure_ascii=False)

    with open('./OCR/result.json', 'w', encoding="UTF-8") as f:
        f.write(json_data)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', required=True, help='path to image_folder which contains text images', default='OCR/content/crop_img/')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation", default='OCR/saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111/best_accuracy.pth')
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS',default='TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet',default='ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM',default='BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn',default='Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    opt = parser.parse_args()

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    
    detecttext()    #문자검출
    demo(opt)       #문자인식
    makedict()      #결과저장