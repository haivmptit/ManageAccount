import base64
import cv2, io
# import argparse
import numpy as np
from PIL import Image
from insightface.model_zoo import get_model
from numpy.linalg import norm
# from server.face_model import FaceModel
from mtcnn.mtcnn import MTCNN
from server import face_preprocess

# parser = argparse.ArgumentParser(description='face model test')
# parser.add_argument('--image-size', default='112,112', help='')
# parser.add_argument('--model', default="./model/model, 0", help='path to load model.')
# parser.add_argument('--ga-model', default="./model/model, 0", help='path to load model.')
# parser.add_argument('--gpu', default=0, type=int, help='gpu id')
# parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
# parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
# parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
# args = parser.parse_args()
# model = FaceModel(args)  # init model detect-face
face = get_model('arcface_r100_v1')  # init model get_emmberding_face
ctx_id = -1
face.prepare(ctx_id=ctx_id)
detector = MTCNN()

def get_face(img):
    faces = detector.detect_faces(img)
    if len(faces) != 1:
        return None
    else:
        box = faces[0]['box']
        x1, y1, width, height = faces[0]['box']

        box[2] = x1 + width
        box[3] = y1 + height
        keypoints = faces[0]['keypoints']
        left_eye = keypoints['left_eye']
        right_eye = keypoints['right_eye']
        nose = keypoints['nose']
        mouth_left = keypoints['mouth_left']
        mouth_right = keypoints['mouth_right']
        points = np.asarray([left_eye, right_eye, nose, mouth_left, mouth_right])
        face_extracted = face_preprocess.preprocess(img, box, points, image_size='112,112')
        return face_extracted


def get_emberding(face_aligned):
    return face.get_embedding(face_aligned).flatten()


def similarity_two_face(img1_base64, img2_base64):
    if img1_base64 is None or img2_base64 is None:
        note = "Loi gui anh, vui long gui lai!"
        return 0, note
    else:
        try:
            img1 = base64.b64decode(str(img1_base64))
            img1 = Image.open(io.BytesIO(img1))
            # img1 = img1.convert('RGB')
            img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_BGR2RGB)
            face1=get_face(img1)

            img2 = base64.b64decode(str(img2_base64))
            img2 = Image.open(io.BytesIO(img2))
            # img2 = img2.convert('RGB')
            img2 = cv2.cvtColor(np.array(img2), cv2.COLOR_BGR2RGB)
            # face2 = model.get_input(img2)
            face2 = get_face(img2)

            note = ""
            if face1 is None or face2 is None:
                note += "Mỗi ảnh phải chứa duy nhất một khuôn mặt. Vui lòng kiểm tra lại!"
            if note != "":
                return 0, note
            else:
                emb1 = face.get_embedding(face1).flatten()
                print(emb1)
                emb2 = face.get_embedding(face2).flatten()
                sim = np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))
                return 1, round(float(sim),2)
        except Exception as e:
            print("Error: ", e)
            note = "Lỗi xử lí!"
            return 0, note
