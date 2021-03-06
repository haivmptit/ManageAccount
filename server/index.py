# start import lib
import os, jwt, json, time, requests, flask, base64, io, cv2, numpy as np
from PIL import Image
from datetime import datetime, timedelta, date
from flask import Flask, request, redirect, url_for, jsonify, send_file, Response, session
from flask_cors import CORS
# from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
from functools import wraps
from server import processing
from gevent.pywsgi import WSGIServer
from server.processing import get_face, get_emberding
from server.connectDB import connection
from numpy.linalg import norm

UPLOAD_FOLDER = './static/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

# config server
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# handle http
@app.route("/huyen")
def index1():
    return "Hello REVA!"


@app.route("/compare_two_image", methods=['POST'])
def compare():
    request_json = request.get_json()
    img1_base64 = request_json.get('img1_base64')
    img2_base64 = request_json.get('img2_base64')
    ip_address = flask.request.remote_addr
    print("Da nhan data tu client: " + ip_address + " ,luc " + str(datetime.now()))
    status, result = processing.similarity_two_face(img1_base64, img2_base64)
    data = {
        'status': status,
        'result': str(result),
    }
    print("Result: " + str(data))

    return jsonify({'status': data['status'], 'result': data['result']})


@app.route("/sign_up", methods=['POST'])
def sign_up():
    request_json = request.get_json()
    imgBase64 = request_json.get('imgBase64')
    username = request_json.get('username')
    name = request_json.get('name')
    phone = request_json.get('phone')
    email = request_json.get('email')
    sex = request_json.get('sex')
    ip_address = flask.request.remote_addr
    print("Yêu cầu: Đăng kí tài khoản.")
    print("Đã nhận từ: " + ip_address + " ,luc " + str(datetime.now()))
    print("Dữ liệu nhận được: " + "username: " + username + ", name: " + name + ", phone: " + phone + ", email: " + email + ", sex: " + sex)
    if imgBase64 is None:
        note = "Loi gui anh, vui long gui lai!"
        return jsonify({'status': 0, 'result': note})
    else:
        try:
            img = base64.b64decode(str(imgBase64))
            img = Image.open(io.BytesIO(img))
            img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
            face = get_face(img)
            note = ""
            if face is None:
                note += "Ảnh đăng kí phải chứa duy nhất một khuôn mặt. Vui lòng kiểm tra lại!"
            if note != "":
                return jsonify({'status': 0, 'result': note})
            else:
                face_emberding = get_emberding(face)
                face_str = ""
                for i in range(len(face_emberding)):
                    face_str += str(face_emberding[i]) + " "
                connect = connection()
                cur = connect.cursor()
                select = " SELECT * FROM person WHERE username = '%s' " % (username)
                cur.execute(select)
                data = cur.fetchall()
                if (len(data) > 0):
                    note = "Tên đăng nhập đã tồn tại!!!"
                    return jsonify({'status': 0, 'result': note})
                else:
                    insert = "INSERT INTO person VALUES('%s', '%s', '%s', '%s', '%s', '%s','%s') " % (
                        username, name, phone, email, sex, face_str, str(imgBase64))
                    cur.execute(insert)
                    connect.commit()

                return jsonify({'status': 1})
        except Exception as e:
            print("Error: ", e)
            note = "Lỗi xử lí!"
            return jsonify({'status': 0, 'result': note})


@app.route("/login", methods=['POST'])
def login():
    request_json = request.get_json()
    imgBase64 = request_json.get('imgBase64')
    # print(len(imgBase64))
    username = request_json.get('username')
    ip_address = flask.request.remote_addr
    print("Yêu cầu: Đăng nhập.")
    print("Đã nhận từ: " + ip_address + " ,luc " + str(datetime.now()))
    print("Dữ liệu nhận được: " + "username: " + username)
    print("------------------------------------------------------------------------------------------------------------")
    if imgBase64 is None:
        note = "Loi gui anh, vui long gui lai!"
        return jsonify({'status': 0, 'result': note})
    else:
        try:
            connect = connection()
            cur = connect.cursor()
            select = " SELECT * FROM person WHERE username = '%s' " % (username)
            cur.execute(select)
            data = cur.fetchall()
            if (len(data) == 0):  # kiem tra ten dang nhap
                note = "Tên đăng nhập không tồn tại!!!"
                return jsonify({'status': 0, 'result': note})
            else:
                for row in data:  # lay thong tin user
                    username = row[0]
                    name = row[1]
                    phone = row[2]
                    email = row[3]
                    sex = row[4]
                    face_str = row[5]
                    face_base64 = row[6]
                face_str = face_str.strip()
                face_str = face_str.split(" ")
                face_signup = np.array(face_str, dtype=float)
                # Kiem tra thong tin khuon mat dang nhap
                img = base64.b64decode(str(imgBase64))
                img = Image.open(io.BytesIO(img))
                img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
                face = get_face(img)
                note = ""
                if face is None:
                    note += "Ảnh đăng nhập phải chứa duy nhất một khuôn mặt. Vui lòng kiểm tra lại!"
                    return jsonify({'status': 0, 'result': note})
                else:
                    face_login = get_emberding(face)
                    sim = np.dot(face_signup, face_login) / (norm(face_signup) * norm(face_login))
                    print("Độ tương đồng giữa 2 khuôn mặt là: ", sim)
                    if (sim > 0.5):
                        return jsonify({'status': 1, 'username': username,
                                        'name': name, 'phone': phone,
                                        'email': email, 'sex': sex, 'face_base64': face_base64})
                    else:
                        note = "Khuôn mặt không khớp với tên đăng nhập, bạn vui lòng kiểm tra lại!"
                        return jsonify({'status': 0, 'result': note})
        except Exception as e:
            print("Error: ", e)
            note = "Lỗi xử lí!"
            return jsonify({'status': 0, 'result': note})


@app.route("/return_profile", methods=['POST'])
def return_profile():
    request_json = request.get_json()
    username = request_json.get('username')
    ip_address = flask.request.remote_addr
    print("Yêu cầu: Lấy thông tin cá nhân.")
    print("Đã nhận từ: " + ip_address + " ,luc " + str(datetime.now()))
    print("Dữ liệu nhận được: " + "username: " + username)
    print("-----------------------------------------------------------------------------------------------------------")
    try:
        connect = connection()
        cur = connect.cursor()
        select = " SELECT * FROM person WHERE username = '%s' " % (username)
        cur.execute(select)
        data = cur.fetchall()

        for row in data:  # lay thong tin user
            username = row[0]
            name = row[1]
            phone = row[2]
            email = row[3]
            sex = row[4]
            face_base64 = row[6]
        return jsonify({'status': 1, 'username': username,
                        'name': name, 'phone': phone,
                        'email': email, 'sex': sex, 'face_base64': face_base64})
    except Exception as e:
        print("Error: ", e)
        note = "Lỗi xử lí!"
        return jsonify({'status': 0, 'result': note})


@app.route("/edit_profile", methods=['POST'])
def edit_profile():
    request_json = request.get_json()
    username = request_json.get('username')
    name = request_json.get('name')
    phone = request_json.get('phone')
    email = request_json.get('email')
    sex = request_json.get('sex')
    ip_address = flask.request.remote_addr
    print("Yêu cầu: Chỉnh sửa thông tin cá nhân.")
    print("Đã nhận từ: " + ip_address + " ,luc " + str(datetime.now()))
    print("Dữ liệu nhận được: " + "username: " + username + ", name: " + name + ", phone: " + phone + ", email: " + email + ", sex: " + sex)
    print("------------------------------------------------------------------------------------------------------------")
    try:
        connect = connection()
        cur = connect.cursor()
        update = "UPDATE person SET name = '%s', phone = '%s', email = '%s', sex = '%s' WHERE  username = '%s' " % (
        name, phone, email, sex, username)
        cur.execute(update)
        connect.commit()
        return jsonify({'status': 1})
    except:
        return jsonify({'status': 0, 'result': "Lỗi server!!!"})

@app.route("/change_face", methods=['POST'])
def change_face():
    request_json = request.get_json()
    imgBase64 = request_json.get('imgBase64')
    username = request_json.get('username')
    ip_address = flask.request.remote_addr
    print("Yêu cầu: Thay đổi khuôn mặt đăng nhập.")
    print("Đã nhận từ: " + ip_address + " ,luc " + str(datetime.now()))
    print("Dữ liệu nhận được: " + "username: " + username)
    print("------------------------------------------------------------------------------------------------------------")
    if imgBase64 is None:
        note = "Loi gui anh, vui long gui lai!"
        return jsonify({'status': 0, 'result': note})
    else:
        try:
            img = base64.b64decode(str(imgBase64))
            img = Image.open(io.BytesIO(img))
            img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
            face = get_face(img)
            note = ""
            if face is None:
                note += "Ảnh cập nhật phải chứa duy nhất một khuôn mặt. Vui lòng kiểm tra lại!"
            if note != "":
                return jsonify({'status': 0, 'result': note})
            else:
                face_emberding = get_emberding(face)
                face_str = ""
                for i in range(len(face_emberding)):
                    face_str += str(face_emberding[i]) + " "

                connect = connection()
                cur = connect.cursor()
                update = "UPDATE person SET face_emberding = '%s',face_base64 = '%s' WHERE  username = '%s' " % (
                    face_str, str(imgBase64), username)

                cur.execute(update)
                connect.commit()

                return jsonify({'status': 1})
        except Exception as e:
            print("Error: ", e)
            note = "Lỗi xử lí!"
            return jsonify({'status': 0, 'result': note})
# def token_required(f):
#     @wraps(f)
#     def decorated(*args, **kwargs):
#         token = request.args.get('token')  # http://0.0.0.0:5002/route?token=.....
#         if not token:
#             return jsonify({'message': 'Token is missing!'}), 403
#         try:
#             data = jwt.decode(token, SECRECT, algorithms=['HS256'])
#         except:
#             return jsonify({'message': 'Token is invalid!'}), 403
#         return f(*args, **kwargs)
#
#     return decorated


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


if __name__ == '__main__':
    # app.run(host='0.0.0.0', debug=False, port=5000)
    http_server = WSGIServer(('0.0.0.0', 80), app)
    http_server.serve_forever()
