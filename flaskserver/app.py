from flask import Flask, request, jsonify, json, send_from_directory
from flask_cors import cross_origin
import os

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello Flask Worldss!!'

@app.route('/api/ocrImageUpload', methods=['GET', 'POST'])
@cross_origin()
def ocrImageUpload():
    json_values = []
    filename=""
    if request.method == 'POST':
        request_data = request.data
        f = request.files['files']
        filename = f.filename
        # f.save("./OCR/test_img/maternity/test.jpg")
        # run OCR
        print("run")
        os.system("python ./OCR/demo.py --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --image_folder OCR/content/crop_img/ --saved_model OCR/saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111/best_accuracy.pth")
        
        print("..")
        if os.path.isfile('./OCR/result.json'):
            with open('./OCR/result.json','r',encoding='utf-8') as resultfile:
                json_data = json.load(resultfile)
                json_values = list(json_data.values())
            os.remove("./OCR/result.json")
            os.remove("./OCR/content/images/test.jpg")
            for file in os.scandir("./OCR/content/crop_img/"):
                os.remove(file.path)
        
    return [filename,json_values]

@app.route('/api/ocrGetImage/<path:subpath>')
def download_File(subpath):
	return send_from_directory('./', subpath, as_attachment=True)
	# return send_file(subpath, as_attachment=True)

if __name__ == '__main__':
    # app.run(host='0.0.0.0')
    app.run(host='0.0.0.0', port=5001, debug=True)	

