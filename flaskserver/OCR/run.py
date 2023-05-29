# import json
# from .demo import demo

# if __name__ == '__main__':
#     p="/home/com_3/workspace/dockerfiles/OCR/"
#     opt = {
#         'Transformation':'TPS',
#         'FeatureExtraction':'ResNet',
#         'SequenceModeling':'BiLSTM',
#         'Prediction':'Attn',
#         'image_folder':f'{p}content/crop_img/',
#         'saved_model': f'{p}saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111/best_accuracy.pth'
#     }
#     demo(opt)

# dict_data = {
#     "모돈번호": "7619-2",
#     "출생일(년)": "21",
#     "출생일(월)": "6",
#     "출생일(일)": "25",
#     "구입일(년)": "22",
#     "구입일(월)": "8",
#     "구입일(일)": "27",
#     "초발정일(년)": "23",
#     "초발정일(월)": "7",
#     "초발정일(일)": "14",
#     "교배일(월)": "7",
#     "교배일(일)": "28",
#     "1차 웅돈번호": "2937-6",
#     "2차 웅돈번호": "5412-4",
#     "재발확인일(월)": "9",
#     "재발확인일(일)": "13",
#     "분만예정일(월)": "10",
#     "분만예정일(일)": "19",
#     "백신1(월)": "8",
#     "백신1(일)": "16",
#     "백신2(월)": "2",
#     "백신2(일)": "12",
#     "백신3(월)": "7",
#     "백신3(일)": "14",
#     "백신4(월)": "5",
#     "백신4(일)": "29"
# }

# json_data = json.dumps(dict_data, indent=1, ensure_ascii=False)

# with open('./OCR/result.json', 'w', encoding="UTF-8") as f:
#     f.write(json_data)