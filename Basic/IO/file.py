#  -*- coding:utf-8 -*-  
import sys
import platform
import os
import datetime
import ast
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Button, RadioButtons
import pandas as pd
import random
from .database import MySqlHandler

my_folder = ["D:\\Hui Lei\\OS_Autoplan", "D:\\Hui Lei\\VueJsSite\\src", "D:\\Downloads\\Cad-master\\"]


def filename_csv(series):
    """Path of storaging file
    :param series: 文件序号
    """
    store_path = 'D:/%s.csv'
    return store_path % series


def ext(file):
    return file.split('.')[-1]


@property
def ON_WINDOWS():
    if platform.system() == "Windows":
        return True
    else:
        return False


def get_desktop():
    if ON_WINDOWS:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                             'Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders', )
        return winreg.QueryValueEx(key, "Desktop")[0]
    else:
        return ""


def cur_dir():
    # 获取脚本路径
    path = sys.path[0]
    # 判断为脚本文件还是py2exe编译后的文件，如果是脚本文件，则返回的是脚本的目录，如果是py2exe编译后的文件，则返回的是编译后的文件路径
    if os.path.isdir(path):
        return path
    elif os.path.isfile(path):
        return os.path.dirname(path)


def get_direct_files(dir):
    return os.listdir(dir)


def get_project_files(dir):
    for root, dirs, files in os.walk(dir):
        # print(files)
        if root.__contains__("packages"):
            continue
        if root.__contains__("WebApi"):
            continue
        if root.__contains__("ObjectScriptingExtensions"):
            continue
        if root.startswith(dir + os.sep + 'adodb5'):
            continue
        if root.startswith(dir + os.sep + 'core\PHPExcel'):
            continue
        if root.startswith(dir + os.sep + 'ext\editor'):
            continue
        for file in files:
            if file in ['PHPExcel.php', 'jquery-1.5.2.js', 'jquery-ui.js', 'jquery-ui.css', 'timer.js']:
                continue
            if file.__contains__("Designer.cs"):
                continue
            if file.__contains__("Temporary"):
                continue
            ext = file.split('.')
            ext = ext[-1]
            if ext in ['php', 'css', 'js', 'html', 'py', 'cs', 'rb', 'vue']:
                it_path = root + os.sep + file
                # print(it_path)
                yield it_path


def get_project_lines(dir, idx = -1):
    def fileline(f_path):
        res = 0
        f = open(f_path, "r", 1, "utf8")
        for lines in f:
            if lines.split():
                res += 1
        return res

    all_line = 0
    allfiles = 0

    if dir is None and idx > -1:
        dir = my_folder[idx]

    for file in get_project_files(dir):
        allfiles += 1
        cur_lines = fileline(file)
        all_line += cur_lines
        print(file, cur_lines)
    print(all_line, allfiles)
    return all_line, allfiles


def combine2one(dir, outfile):
    """ Delete blanklines of infile """
    outfp = open(outfile, "w", encoding = "gb18030", errors = "ignore")
    for file in get_project_files(dir):
        infp = open(file, "r", encoding = "gb18030", errors = "ignore")
        lines = infp.readlines()
        for li in lines:
            if li.split():
                outfp.writelines(li)
                infp.close()
    outfp.close()


def matrix2csv(path, matrix):
    for row in matrix:
        list2csv(path, row)


def list2csv(path, list):
    with open(path, 'w') as file:
        for element in list:
            file.write('%s,' % element)
        file.write('\n')


def obj2file(path, obj, mode = 'w'):
    with open(path, mode) as file:
        content = str(obj)
        file.write(content)


def file2obj(path):
    with open(path, 'r') as file:
        content = file.read()
        result = ast.literal_eval(content)
    return result


class CodeLines:
    @classmethod
    def mine(cls):
        all_line = 0
        allfiles = 0

        for path in (cls.path_cad):
            temp = cls.get_project_lines(path)
            all_line += temp[0]
            allfiles += temp[1]
        print('Total lines:', all_line)
        print('File Number:', allfiles)
