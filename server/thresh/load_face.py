import cv2
import os
import numpy as np
from numpy.linalg import norm
from numpy import savez_compressed
from PIL import Image
from os import listdir
from numpy import asarray
from mtcnn.mtcnn import MTCNN
# import argparse
# from server.face_model import FaceModel
from insightface.model_zoo import get_model
from server import face_preprocess
# parser = argparse.ArgumentParser(description='face model test')
# parser.add_argument('--image-size', default='112,112', help='')
# parser.add_argument('--model', default="../model/model, 0", help='path to load model.')
# parser.add_argument('--ga-model', default="../model/model, 0", help='path to load model.')
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
def get_name(name_file):
    s = name_file.split('.')
    return s[0]

def extract_face(file):
    img = Image.open(file)
    img = img.convert('RGB')
    img = asarray(img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # bor di
    # cv2.imshow('rr',img)
    # cv2.waitKey(0)
    faces = detector.detect_faces(img)
    if len(faces)==0:
        cv2.imshow('t', img)
        cv2.waitKey(0)
    # face_extracted = model.get_input(img)
    # print(faces[0]['keypoints'])

    box = faces[0]['box']
    x1, y1, width, height = faces[0]['box']
    box[2] = x1 + width
    box[3] = y1 + height

    # print(box)
    # print(faces[0]['keypoints'])
    keypoints = faces[0]['keypoints']
    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']
    nose = keypoints['nose']
    mouth_left = keypoints['mouth_left']
    mouth_right = keypoints['mouth_right']
    points = np.asarray([left_eye, right_eye, nose, mouth_left, mouth_right])
    # print(point)
    # face_extracted = face_preprocess.preprocess(img, box, point, image_size='112,112')
    face_extracted = face_preprocess.preprocess(img,box, points, image_size='112,112')
    emb = face.get_embedding(face_extracted).flatten()
    return emb

def load_dataset(directory):
    X, y = list(), list()  # X : danh sách chua khuon mat, y: nhan cua moi anh
    # Duyệt các ảnh trong mỗi thư mục
    for subdir in listdir(directory):
        print('<3 Loading the image for datasets: %s' % (subdir))
        file = directory + subdir
        face = extract_face(file)
        # print('<3 Loaded the image for datasets: %s' % (subdir))
        X.append(face)
        y.append(get_name(subdir))
    return asarray(X), asarray(y)
def similarity_cos(emb1,emb2): # tính độ tương tự giữa 2 vecto emberding khuôn mặt.
    sim = np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))
    return sim

print("Loading dataset......................")
# cardX, cardId = load_dataset('D:\Đồ án\DataImage\\Nu\card_nu\\')
# savez_compressed('faces-dataset-card.npz',cardX, cardId)
# liveX, liveId = load_dataset('D:\Đồ án\DataImage\\Nu\live_nu\\')
# savez_compressed('faces-dataset-live.npz', liveX, liveId)
# print(liveX.shape, liveId.shape)

liveX1, liveId1 = load_dataset('D:\Đồ án\DataImage\\Nam\card_nam\\')
cardX1, cardId1 = load_dataset('D:\Đồ án\DataImage\\Nam\live_nam\\')

##############
liveX1 = liveX1.tolist()
cardX1 = cardX1.tolist()

# for i in range(np.shape(liveX1)[0]):
#     similarity = round(float(similarity_cos(liveX1[i], cardX1[i])), 2)
#     if similarity < 0.4:
#         print("###Giong nhau nhung nho hon 0.5: ", cardId1[i], similarity)
#     else:
#         print("Giong nhau la: ", similarity)

 # xóa những ảnh nếu độ tương đồng thấp hơn 1 ngưỡng nào đó.

liveX, liveId=[],[]
cardX, cardId = [],[]
directory_card = 'D:\Đồ án\DataImage\\Nam\card_nam\\'
directory_live = 'D:\Đồ án\DataImage\\Nam\live_nam\\'
for i in range(np.shape(liveX1)[0]):
    similarity = round(float(similarity_cos(liveX1[i], cardX1[i])), 2)
    if similarity < 0.4:
        print("###Giong nhau nhung nho hon 0.4: ", cardId1[i], similarity)
        for subdir in listdir(directory_card):
            if cardId1[i]== get_name(subdir) :
                file = directory_card + subdir
                os.remove(file)
                print("Da xoa " + file )
                break
        for subdir in listdir(directory_live):
            if liveId1[i]== get_name(subdir) :
                file = directory_live + subdir
                os.remove(file)
                print("Da xoa " + file )
                break
    else:
        print("Giong nhau la: ", similarity)
        liveX.append(liveX1[i])
        liveId.append(liveId1[i])
        cardX.append(cardX1[i])
        cardId.append(cardId1[i])
print(np.shape(liveX),np.shape(liveId) , np.shape(cardX), np.shape(cardId))
# savez_compressed('faces-dataset-card.npz',cardX, cardId)
# savez_compressed('faces-dataset-live.npz', liveX, liveId)