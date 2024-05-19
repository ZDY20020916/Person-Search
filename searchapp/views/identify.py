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
from io import BytesIO
from utils.imgprocess import read_image
from reid.extractor import getFeature, rank
import numpy as np


'''重识别结果信息表'''
def search_list(request):
    user_id = request.session["info"]["id"]
    image_list = request.session.pop('image_list', None)
    if image_list is None:
        print('None')
        # 无结果返回
        queryset = models.BoxSet.objects.none()
    else:
        # 返回匹配项
        print(image_list)
        queryset = models.BoxSet.objects.filter(id__in=image_list).order_by("-id")
        print(queryset)
    page_object = Pagination(request, queryset, page_size=5)
    context = {
        "queryset": page_object.page_queryset,  # 分完页的数据
        "page_string": page_object.html()  # 生成页码
    }
    return render(request, 'search_list.html', context)


class BoxModelForm(BootStrapModelForm):

    bootstrap_exclude_fields = ["img"]  # 排除不想添加样式的字段

    class Meta:
        model = models.BoxSet
        exclude = ["img_info", "admin", "source_id", "feature"]  # 排除oid和admin后的所有属性（排除订单号和所属用户）
        widgets = {
            "img": forms.FileInput(attrs={"accept": "image/*"}),
            # 限制input file上传类型只能是图片: <input type="file" accept="image/*">
        }


'''上传行人图片'''
def identify(request):
    title = "上传检测框图像"
    req_method = request.method
    if req_method == 'GET':
        form = BoxModelForm()
        return render(request, 'upload_form.html', {"title": title, "form": form})
    form = BoxModelForm(data=request.POST, files=request.FILES)

    if form.is_valid():
        img_object = form.cleaned_data.get('img') or request.FILES.get('img')
        # 图片保存的临时路径
        img_path = os.path.join(settings.MEDIA_ROOT, 'uploads', img_object.name)
        # 确保保存路径存在
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        # 将图片保存到磁盘上
        with open(img_path, 'wb+') as destination:
            for chunk in img_object.chunks():
                destination.write(chunk)
        # 使用保存的图片路径调用 read_image 函数
        boximage = read_image(img_path)
        feature = getFeature(boximage)
        boximage.close()
        boxes = models.BoxSet.objects.all()
        os.chdir(settings.BASE_DIR)
        feature_arrays = []
        for box in boxes:
            feature_name = str(box.feature)
            feature_path = os.path.join('data', feature_name)
            id = box.id
            a = np.load(feature_path)
            feature_arrays.append((id, a))
        # 查找相似行人图片
        id_list = rank(feature, feature_arrays)
        # 将显示列表id插入会话
        print(id_list)
        request.session['image_list'] = id_list
        # 返回根目录
        return redirect('/reid/')
    return render(request, 'upload_form.html', {"form": form, "title": title})


def source_display(request, yid):
    box = models.BoxSet.objects.filter(id=yid).first()
    if not box:
        return render(request, 'error.html', {"msg": "数据不存在"})
    if not box.source_id:
        return render(request, 'error.html', {"msg": "数据库出现致命错误"})
    image_url = box.source_id.processed_img.url if box.source_id.processed_img.url else None
    return JsonResponse({'image_url': image_url})
