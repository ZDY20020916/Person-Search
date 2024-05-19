from django.shortcuts import render, HttpResponse, redirect
from searchapp.utils.pagination import Pagination
from searchapp.utils.drawio import drawbox
from searchapp import models
from searchapp.utils.bootstrap import BootStrapModelForm
from django import forms
from django.http import JsonResponse
from yolov8 import predict

import os
import glob
from django.conf import settings
import json
from django.core.files.base import ContentFile
from io import BytesIO
from reid.extractor import getFeature
import numpy as np
import logging
from django.core.files.storage import default_storage


'''行人信息表'''
def person_list(request):
    user_id = request.session["info"]["id"]
    queryset = models.PersonSet.objects.all().order_by("-id")
    # queryset = models.PersonSet.objects.filter(admin_id=user_id).order_by("-id")
    page_object = Pagination(request, queryset, page_size=5)
    context = {
        "queryset": page_object.page_queryset,  # 分完页的数据
        "page_string": page_object.html()  # 生成页码
    }
    return render(request, 'person_list.html', context)


class PersonModelForm(BootStrapModelForm):

    bootstrap_exclude_fields = ["img"]  # 排除不想添加样式的字段

    class Meta:
        model = models.PersonSet
        exclude = ["img_result", "admin", "img_name", "processed_img"]  # 排除oid和admin后的所有属性（排除订单号和所属用户）
        widgets = {
            "img": forms.FileInput(attrs={"accept": "image/*"}),
            # 限制input file上传类型只能是图片: <input type="file" accept="image/*">
        }


'''上传行人图片'''
def person_add(request):
    title = "上传行人图像"
    req_method = request.method
    if req_method == 'GET':
        form = PersonModelForm()
        return render(request, 'upload_form.html', {"title": title, "form": form})
    form = PersonModelForm(data=request.POST, files=request.FILES)

    if form.is_valid():
        # 对于文件：自动保存；
        # 字段 + 上传路径写入到数据库
        # 不用担心重复上传同名文件会被覆盖的情况，它会自动给重名的文件加uuid
        # 固定设置管理员ID (当前登录用户的id)
        form.instance.admin_id = request.session["info"]["id"]

        form.save()
        yid = form.instance.id
        name = str(form.instance.img).split('/')[-1]
        models.PersonSet.objects.filter(id=yid).update(img_name=name)

        # 返回根目录
        return redirect('/')
    return render(request, 'upload_form.html', {"form": form, "title": title})


'''删除行人图像'''
def person_del(request, yid):
    '''在删除数据库数据的同时把对应的图片删除'''
    person = models.PersonSet.objects.filter(id=yid).first()
    boxes = models.BoxSet.objects.filter(source_id=yid)
    if not person:
        return render(request, 'error.html', {"msg": "数据不存在"})
    os.chdir(settings.BASE_DIR)
    p_path = str(os.getcwd())
    #删除original和processed
    img_name = str(person.img)
    processed_img_name = str(person.processed_img)
    original_path = glob.glob(os.path.join('data', img_name))
    processed_path = glob.glob(os.path.join('data', processed_img_name))
    for file in original_path:
        os.remove(file)
    for file in processed_path:
        os.remove(file)
    #删除所有box和feature
    for box in boxes:
        box_name = str(box.img)
        box_path = glob.glob(os.path.join('data', box_name))
        feature_name = str(box.feature)
        feature_path = glob.glob(os.path.join('data', feature_name))
        for file in box_path:
            os.remove(file)
        for file in feature_path:
            os.remove(file)
    models.PersonSet.objects.filter(id=yid).delete()
    models.BoxSet.objects.filter(source_id=yid).delete()
    return redirect('/')


'''行人检测'''
def person_detect(request, yid):
    person = models.PersonSet.objects.filter(id=yid).first()
    if not person:
        return render(request, 'error.html', {"msg": "数据不存在"})
    try:
        img_path = os.path.join(settings.MEDIA_ROOT, str(person.img))
        save_name = person.img_name.rsplit('.', 1)[0]
        result, processed_img, croppedboxes = predict.detect_original(img_path)
        person.img_result = result
        person.save()
        # 储存带检测框的原图像
        if processed_img.format is None:
            processed_img.format = 'PNG'
        img_byte_arr = BytesIO()
        processed_img.save(img_byte_arr, format=processed_img.format)
        processed_img_name = f"{save_name}.png"
        img_file = ContentFile(img_byte_arr.getvalue(), name=processed_img_name)
        person.processed_img.save(processed_img_name, img_file, save=True)
        # 储存所有检测框
        for idx, croppedbox in enumerate(croppedboxes):
            if croppedbox.format is None:
                croppedbox.format = 'PNG'
            # 检测盒
            box_byte_arr = BytesIO()
            croppedbox.save(box_byte_arr, format=croppedbox.format)
            box_name = f"{idx}_{save_name}.png"
            box_file = ContentFile(box_byte_arr.getvalue(), name=box_name)
            print('ok1')
            # 特征.npy文件
            feature = getFeature(croppedbox)
            buffer = BytesIO()
            np.save(buffer, feature)
            buffer.seek(0)
            npy_name = f"{idx}_weights_of_{save_name}.npy"
            npy_content_file = ContentFile(buffer.getvalue(), name=npy_name)
            print('ok2')
            # 创建实例，并保存
            print('ok3')
            box_set_instance = models.BoxSet()
            print('ok4')
            box_set_instance.img.save(box_name, box_file, save=False)
            box_set_instance.img_info = person.img_name
            box_set_instance.source_id = person
            box_set_instance.feature.save(npy_name, npy_content_file, save=False)
            print(person.admin)
            box_set_instance.admin = person.admin
            box_set_instance.save()
            print('hahaha')
        os.chdir(settings.BASE_DIR)  # 改变工作目录
        return JsonResponse({"status": True})
    except Exception as e:
        os.chdir(settings.BASE_DIR)  # 改变工作目录
        logging.error(f"Detection failed due to: {e}")
        return JsonResponse({"status": False, "msg": "检测失败，数据异常"})


def person_display(request, yid):
    person = models.PersonSet.objects.filter(id=yid).first()
    if not person:
        return render(request, 'error.html', {"msg": "数据不存在"})
    if not person.img_result:
        return render(request, 'error.html', {"msg": "该图片尚未检测"})
    image_url = person.processed_img.url if person.processed_img else None
    return JsonResponse({'image_url': image_url})
