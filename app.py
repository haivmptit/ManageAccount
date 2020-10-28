# start import lib
# import flask
from flask import Flask, request, jsonify
from server import processing
# from server import file_import
# config server
app = Flask(__name__)

# handle http
@app.route("/", methods=['POST','GET'])
def index1():
    dem =""
    try:
        import base64
        dem="import base64"
        import cv2 #loi
        dem = "cv2"
        import io
        dem = "io"
        import numpy as np
        dem = "import numpy as np"
        from PIL import Image
        dem = "from PIL import Image"
        # from insightface.model_zoo import get_model
        # dem = " from insightface.model_zoo import get_model"
        from numpy.linalg import norm
        dem = " from numpy.linalg import norm"
        from mtcnn.mtcnn import MTCNN # die
        dem = " from mtcnn.mtcnn import MTCNN"
        from server import face_preprocess # co cv2
        dem="from server import face_preprocess"
        # face = get_model('arcface_r100_v1')  # init model get_emmberding_face
        dem = "get_model"
        ctx_id = -1
        # face.prepare(ctx_id=ctx_id)
        dem = "face.prepare(ctx_id=ctx_id)"
        # detector = MTCNN()
        dem = "detector = MTCNN()"
        return dem + "OK"
    except:
        return dem



@app.route("/compare_two_image", methods=['POST','GET'])
def compare():
    request_json = request.get_json()
    img1_base64 = request_json.get('img1_base64')
    img2_base64 = request_json.get('img2_base64')
    status, result = processing.similarity_two_face(img1_base64, img2_base64)
    data = {
        'status': status,
        'result': str(result),
    }

    return jsonify({'status': data['status'], 'result': data['result']})

if __name__ == '__main__':
    app.run()
# if __name__ == '__main__':
#     # app.run(host='0.0.0.0', debug=False, port=5000)
#     http_server = WSGIServer(('0.0.0.0', 80), app)
#     http_server.serve_forever()
