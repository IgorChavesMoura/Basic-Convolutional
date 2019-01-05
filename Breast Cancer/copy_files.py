# -*- coding: utf-8 -*-

from shutil import copyfile,copy2

import os


def copy_file(filename,filedir,destdir):
    
    if not os.path.exists(destdir):
        os.makedirs(destdir)

    copyfile(filedir + '/' + filename, destdir + '/' + filename)