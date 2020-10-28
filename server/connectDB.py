import psycopg2
import numpy as np
def connection():
    conn = psycopg2.connect(
        database="demo1",
        user="postgres",
        password="songuyento",
        host="localhost",
        port="5432")
    return conn
# username = 'haivm'
# select = " SELECT * FROM person WHERE username = '%s' " %(username)
# insert = "INSERT INTO person VALUES('dungnd1', 'dung', '23434', 'sdfsdf', 'nu', 'sdfsdf') "
# con = connection()
# cur = con.cursor()
# cur.execute(insert)
# con.commit()

# data = cur.fetchall()
# print(len(data))
# for row in data:
#     # printing the columns
#     print('username: ', row[0])
#     print('name: ', row[1])
# s = "2.3 4.5 6.7"
# s= s.strip()
# a= s.split(" ")

# print(a)
# print (np.array(a, dtype=float))