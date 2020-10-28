import requests
import base64
from datetime import datetime
with open("../Image/lanhcmt.JPG", "rb") as img_file:
    img1_base64 = base64.b64encode(img_file.read())
with open("../Image/Lan_Ngoc.jpg", "rb") as img_file:
    img2_base64 = base64.b64encode(img_file.read())
# with open("D:\Đồ án\DataImage\\Nam\live_nam\\25975.jpg", "rb") as img_file:
#     img1_base64 = base64.b64encode(img_file.read())
# with open("D:\Đồ án\DataImage\\Nam\card_nam\\25975.JPG", "rb") as img_file:
#     img2_base64 = base64.b64encode(img_file.read())
data ={'img1_base64': img1_base64.decode('utf-8'),
       'img2_base64': img2_base64.decode('utf-8')}
# print(data)
time1 = datetime.now()
# response = requests.post('https://serene-basin-15155.herokuapp.com/compare_two_image', json=data)
response = requests.post('http://localhost:80/compare_two_image', json=data)
re = response.json()
print(re['status'])
print(re['result'])
time2 = datetime.now()
print(time1)
print(time2)

# print(response.json())