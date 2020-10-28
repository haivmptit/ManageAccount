import numpy as np
import random
from numpy.linalg import norm
from numpy import load


def similarity_cos(emb1, emb2):  # tính độ tương tự giữa 2 vecto emberding khuôn mặt.
    sim = np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))
    return sim


data = load('../thresh/faces-dataset-live.npz')
liveX, liveId = data['arr_0'], data['arr_1']
data = load('../thresh/faces-dataset-card.npz')
cardX, cardId = data['arr_0'], data['arr_1']
print(np.shape(liveX)[0])
print(np.shape(cardX)[0])
print(np.shape(liveId))
# print(liveId)
liveX = liveX.tolist()
liveId = liveId.tolist()
cardX = cardX.tolist()
cardId = cardId.tolist()
number_img = len(liveX)
for i in range(len(liveX)):
    j = random.randint(0, number_img - 1)
    liveX.append(liveX[j])
    liveId.append(liveId[j])

    j = random.randint(0, number_img - 1)
    cardX.append(cardX[j])
    cardId.append(cardId[j])
print(np.shape(liveX)[0])
for i in range(np.shape(liveX)[0]):
    similarity = round(float(similarity_cos(liveX[i], cardX[i])), 2)
    if (liveId[i] == cardId[i]):
        print("Giong: ", similarity)
        if similarity < 0.4:
            print("Giong nhau nhung nho hon 0.5: ", cardId[i])
    else:
        print("Khac: ", similarity)
pairs = len(liveX)
step_thresh = 0.02
while (step_thresh < 1):
    same = 0
    for i in range(pairs):
        if (similarity_cos(liveX[i], cardX[i]) >= step_thresh):
            predict = 1
        else:
            predict = 0
        if (liveId[i] == cardId[i] and predict == 1):
            same += 1
        if (liveId[i] != cardId[i] and predict == 0):
            same += 1
    print("Voi nguong %f thi du doan dung la %f: " % (step_thresh, round(float(same / pairs) * 100, 2)))
    step_thresh += 0.02
