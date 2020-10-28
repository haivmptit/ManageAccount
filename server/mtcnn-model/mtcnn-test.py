import cv2
import argparse
import numpy as np
from insightfacee.model_zoo import get_model
from deploy.face_model import FaceModel
from mtcnn.mtcnn import MTCNN
from PIL import Image
parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='C:\AI\Code\insightface-master\gender-age\model\model,0', help='path to load model.')
parser.add_argument('--ga-model', default='C:\AI\Code\insightface-master\gender-age\model\model,0', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

model = FaceModel(args) # init model detect-face
img1 = cv2.imread('C:\AI\Code\Classify-age\ClassifyAge\Test_image\\haichandung.jpg')
img1 = model.get_input(img1) #  Get aligned1
img1 = np.transpose(img1, (1,2,0))

img2 = cv2.imread('C:\AI\Code\Classify-age\ClassifyAge\Test_image\\haivu.jpg')
img2 = model.get_input(img2) # Get aligned2
img2 = np.transpose(img2, (1,2,0))
print(np.shape(img2))
face =get_model('arcface_r100_v1') # init model get_emmberding_face
ctx_id = -1
face.prepare(ctx_id = ctx_id)
# Dùng face.get_embedding(img)
emb1 = face.get_embedding(img1).flatten()
emb2 = face.get_embedding(img2).flatten()
from numpy.linalg import norm
print(norm(emb1))
sim = np.dot(emb1, emb2)/(norm(emb1)*norm(emb2))
print("Similarity 1 là: ",sim)
#Dùng face.compute_sim(img1, img2)
# print("Similarity 2 là: ", face.compute_sim(img1,img2))