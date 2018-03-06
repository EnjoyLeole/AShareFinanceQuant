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


class LookOver:
    def __init__(self):
        self.my_root = ""
        self.cur_root = ""
        self.cur_files = ""
        self.cur_i = 0
        self.delete_list = []
        self.lena = None
        # self.plt=None

    @property
    def cur_file(self):
        return self.cur_root + "\\" + self.cur_files[self.cur_i]

    def iterator(self, handler):
        def iterator():
            for root, dirs, files in os.walk(self.my_root):
                # if len(files):
                handler(root, dirs, files)

        return iterator

    def run(self):
        raise NotImplementedError


class Naming:
    def __init__(self, my_root = None):
        super().__init__()
        self.my_root = "D:\\360data\\重要数据\\3e088979637ca3" if my_root is None else my_root
        self.prefix = "_x"

        self.rand = random.sample(range(0, 297281), 200000)
        self.rand_count = 0

    def get_rand(self):
        self.rand_count += 1
        return self.rand[self.rand_count]

    def run(self):
        d_count = 0
        f_count = 0
        no_drill = True

        def change_save(path, name, ifdir):
            old_path = path + "\\" + name
            if name[0:2] != self.prefix:
                new_name = "%s %i" % (self.prefix, self.get_rand())
                new_path = path + "\\" + new_name
                if no_drill:
                    os.rename(old_path, new_path)
                log.append([old_path, name, new_path, new_name, ifdir, datetime.datetime.now()])
                print("%s ::  %s -> %s %s" % (path, name, new_name, ifdir))

        try:
            for root, dirs, files in os.walk(self.my_root):
                log = []
                for dir in dirs:
                    change_save(root, dir, True)
                    d_count += 1
                for file in files:
                    change_save(root, file, False)
                    f_count += 1
                df = pd.DataFrame(log, columns = ["old_path", "old_name", "new_path", "new_name", "if_dir", 'datetime'])
                ms = MySqlHandler("szlib")
                ms.table_save(df, "log_file", index_label = ["newpath"])
        finally:
            print(d_count, f_count)

    @staticmethod
    def restore_names():
        ms = MySqlHandler("szlib")

        def files():
            df = ms.table_read("log_file")
            # df = pd.DataFrame()
            df = df.sort_values('datetime', ascending = False)
            for i, row in df.iterrows():
                try:
                    os.rename(row.new_path, row.old_path)
                    print(i, row.new_path, row.old_path)
                except Exception as e:
                    print(e)

        #
        # def dirs():
        #     df = ms.table_read("log_dir")
        #     for i, row in df.iterrows():
        #         try:
        #             os.rename(row.new_path, row.old_path)
        #             print(row.new_path, row.old_path)
        #         except Exception as e:
        #             print(e)

        # dirs()
        files()


class Audit(LookOver):
    my_root = ""

    def run(self):
        pass

    def handle(self, root, dirs, files):
        ext = files[0].split('.')[-1]
        if ext == "jpg" or ext == "png":
            self.cur_files = files
            self.cur_root = root
            self.cur_i = 0
            self._show_pic(self.cur_file)

    def _show_pic(self, file, file1 = None):
        self.lena = mpimg.imread(file)
        plt.figure(1)
        ax1 = plt.figure(1).add_subplot(111)
        ax1.imshow(self.lena)
        ax1.axis('off')
        name = self.cur_root.split('\\')[-1]
        plt.text(-2, 3, name)
        plt.figure(1).canvas.draw()

        posi = [0.1, 0.03, 0.2, 0.1]

        buttonax1 = plt.axes(posi)
        button1 = Button(buttonax1, "Next")
        button1.on_clicked(self.next)

        posi[0] += 0.2
        buttonax2 = plt.axes(posi)
        button2 = Button(buttonax2, "Delete")
        button2.on_clicked(self.delete)
        posi[0] += 0.2
        buttonax = plt.axes(posi)
        button = Button(buttonax, "Next Pic")
        button.on_clicked(self.next_pic)
        posi[0] += 0.2
        buttonax4 = plt.axes(posi)
        button4 = Button(buttonax4, "Open")
        button4.on_clicked(self.open)

        plt.axis('off')

        plt.show()

    def next(self, event):
        plt.close()

    def next_pic(self, event):
        self.cur_i += 1
        self._show_pic(self.cur_file)

    def delete(self, event):
        self.lena = None
        plt.close()
        name = self.cur_root.split('_')[-1]
        print(name, os.path.dirname(self.cur_root))
        p = self.cur_root.split('\\')
        for root, dirs, files in os.walk(self.cur_root, topdown = False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.cur_root)
        os.remove(os.path.dirname(self.cur_root) + "\\%s.zip" % p[-1])

    def open(self, event):
        from PIL import Image
        self.cur_i += 1
        img = Image.open(self.cur_file)
        img.show()
        # os.startfile(self.cur_root)
        # os.system("explorer.exe %s" % self.cur_root)
