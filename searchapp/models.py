from django.db import models
from django.utils.timezone import now
import datetime
import django
from django.db.models import JSONField
import numpy as np
from django.core.files.base import ContentFile
from io import BytesIO


class Admin(models.Model):
    # 用户
    username = models.CharField(verbose_name="用户名", max_length=32)
    password = models.CharField(verbose_name="密码", max_length=64)

    role_choices = (
        (1, "用户"),
        (2, "管理员")
    )
    role = models.SmallIntegerField(verbose_name="角色", choices=role_choices, default=1)

    name = models.CharField(verbose_name="姓名", max_length=16)
    age = models.IntegerField(verbose_name="年龄")
    # 在django中做的约束
    gender_choices = (
        (1, "男"),
        (2, "女"),
    )
    gender = models.SmallIntegerField(verbose_name="性别", choices=gender_choices)
    mobile = models.CharField(verbose_name="手机号", max_length=32)
    create_time = models.DateField(verbose_name="创建时间", default=now)

    def __str__(self):  # 输出对象时显示username
        return self.username


def default_img_result():
    return [[]]


class PersonSet(models.Model):
    # 行人信息表
    # 本质上数据库也是CharField，会自动保存数据。  upload_to='original_imgs/' 表示文件会自动保存在data目录下的original_imgs目录中
    img = models.FileField(verbose_name="行人图片", max_length=128, upload_to='original_imgs/')

    # img_result decimal(10,2) 小数
    img_result = JSONField(verbose_name="识别结果", default=default_img_result)

    img_name = models.CharField(verbose_name="图片名称", max_length=128)

    processed_img = models.FileField(verbose_name="检测图片", max_length=128, upload_to='processed_imgs/', null=True)

    # 会在数据库表中自动创建admin_id字段
    admin = models.ForeignKey(verbose_name="用户", to="Admin", to_field="id", on_delete=models.CASCADE)


class BoxSet(models.Model):
    # 行人信息表
    # 本质上数据库也是CharField，会自动保存数据。  upload_to='boxes_imgs/' 表示文件会自动保存在data目录下的boxes_imgs目录中
    img = models.FileField(verbose_name="检测框", max_length=128, upload_to='boxes_imgs/')  # 路径

    img_info = models.CharField(verbose_name="图片信息", max_length=128, default='')

    # 来源图片id
    source_id = models.ForeignKey(verbose_name="用户", to="PersonSet", to_field="id", on_delete=models.CASCADE,
                                  related_name="boxes", null=True)

    # 特征向量文件
    feature = models.FileField(verbose_name="特征向量", upload_to='npy_files/', null=True)

    # 会在数据库表中自动创建admin_id字段
    admin = models.ForeignKey(verbose_name="用户", to="Admin", to_field="id", on_delete=models.CASCADE)
