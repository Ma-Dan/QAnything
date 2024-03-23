from flask import Flask, request
from flask import render_template
import numpy as np
import base64
import json
import urllib
from BCEmbedding import EmbeddingModel
from BCEmbedding import RerankerModel
from paddleocr import PaddleOCR
import base64
import numpy as np

# 初始化 BCEmbedding 模型
embedding_model = EmbeddingModel(model_name_or_path="BCEmbedding/bce-embedding-base_v1")
reranker_model = RerankerModel(model_name_or_path="BCEmbedding/bce-reranker-base_v1")

# 初始化 PaddleOCR 引擎
ocr_engine = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=False, show_log=False, enable_mkldnn=True)

app = Flask(__name__)

@app.route('/embeddings',methods=['POST'])
def embeddings():
    sentences = request.json['sentences']
    embeddings = embedding_model.encode(sentences)

    return json.dumps({
            'embeddings': embeddings.tolist()
        }, ensure_ascii=False).encode('utf8')

@app.route('/reranker',methods=['POST'])
def reranker():
    query = request.json['query']
    passages = request.json['passages']

    sentence_pairs = [[query, passage] for passage in passages]

    scores = reranker_model.compute_score(sentence_pairs)

    return json.dumps({
            'scores': scores
        }, ensure_ascii=False).encode('utf8')

@app.post("/ocr")
def ocr_request():
    # 获取上传的文件
    input = request.json
    img_file = input['img64']
    height = input['height']
    width = input['width']
    channels = input['channels']

    binary_data = base64.b64decode(img_file)
    img_array = np.frombuffer(binary_data, dtype=np.uint8).reshape((height, width, channels))
    print("shape: {}".format(img_array.shape))

    # 无文件上传，返回错误
    if not img_file:
        return json.dumps({
            'error': 'No file was uploaded.'
        }, ensure_ascii=False).encode('utf8')

    # 调用 PaddleOCR 进行识别
    res = ocr_engine.ocr(img_array)
    print("ocr result: {}".format(res))

    # 返回识别结果
    return json.dumps({
            'results': res
        }, ensure_ascii=False).encode('utf8')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=7861, threaded=False)
