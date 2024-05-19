from django.conf import settings
import hashlib

'''md5加密'''
def md5(data_string):
    # salt = "xxxxxxxxxxxxxxxx"  # 固定值部分
    # obj = hashlib.md5(salt.encode('utf-8'))

    obj = hashlib.md5(settings.SECRET_KEY.encode('utf-8'))
    obj.update(data_string.encode('utf-8'))
    return obj.hexdigest()


if __name__ == "__main__":
    ss = 'th5256854zj@yky'
    print(md5(ss))
