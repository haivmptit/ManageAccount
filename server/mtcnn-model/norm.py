from sklearn.preprocessing import Normalizer
import sklearn
import numpy as np
a=np.asarray([[4,9,25]])
input= Normalizer(norm='l2')
a= input.transform(a).flatten()
print(a)
sum = 0
for i in range(len(a)):
    sum += a[i]*a[i]
print("Sum: ", sum)
b=np.asarray([[4,9,25]])
embedding = sklearn.preprocessing.normalize(b).flatten()
print(embedding)