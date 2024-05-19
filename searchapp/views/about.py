from django.shortcuts import render, HttpResponse, redirect
from searchapp.utils.pagination import Pagination
from searchapp.utils.drawio import drawbox
from searchapp import models
from searchapp.utils.bootstrap import BootStrapModelForm
from django import forms
from django.http import JsonResponse

import os
import glob
from django.conf import settings
import json


def story(request):
    return render(request, "introduction.html")