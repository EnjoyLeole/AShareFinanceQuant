import pip
from subprocess import call


def update_all():
    for dist in pip.get_installed_distributions():
        call("pip install --upgrade " + dist.project_name, shell = True)


pkg_list = 'D:/SoftWare/Developer/Python/pkg.txt'


def save_pkg_list():
    cmd = "pip freeze >%s" % pkg_list
    print(cmd)
    call(cmd)


def restore_pkgs():
    call("pip install -r %s" % pkg_list)
