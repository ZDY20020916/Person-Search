"""PersonSearch URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from searchapp.views import admin, account, meter, detect, identify, about
from django.urls import path, re_path
from django.views.static import serve
from django.conf import settings


urlpatterns = [

    # 配置media
    re_path(r'^data/(?P<path>.*)$', serve, {'document_root': settings.MEDIA_ROOT}, name='data'),

    # 登录
    path('login/', account.login),
    path('logout/', account.logout),  # 注销
    path('image/code/', account.image_code),  # 生成图像验证码

    # 用户
    path('admin/list/', admin.admin_list),
    path('admin/add/', admin.admin_add),
    path('user/add/', admin.user_add),   # 用户注册
    path('admin/<int:aid>/edit/', admin.admin_edit), # 管理员页面
    path('user/<int:aid>/edit/', admin.user_edit),  # 用户更新
    path('admin/del/', admin.admin_del), # 管理员删除用户
    path('admin/<int:aid>/reset/', admin.admin_reset), # 管理员重置密码
    path('user/<int:aid>/reset/', admin.user_reset),  # 用户更新密码

    # 测试用欢迎界面
    # path('', meter.index),  # 欢迎界面

    # 行人检测
    path('', detect.person_list),  # 行人列表（设置为成为默认的启动地址）
    path('person/add/', detect.person_add),                # 上传包含行人的图像
    path('person/<int:yid>/del/', detect.person_del),      # 删除包含行人的图像
    path('person/<int:yid>/detect/', detect.person_detect),    # 行人检测
    path('person/<int:yid>/display/', detect.person_display),   # 检测后的图像显示

    # 行人搜索
    path('reid/', identify.search_list),   # 重识别结果列表，最多显示5个，后期改为用户自行设定
    path('reid/identify/', identify.identify),   # 上传包含行人的图像
    path('reid/<int:yid>/display/', identify.source_display),   # 识别后的来源图像显示

    # 关于
    path('about/', about.story),

]
