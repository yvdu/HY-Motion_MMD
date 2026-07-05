"""mobupy: 打开 ASCII FBX 另存为 binary FBX(供 Blender 导入)。"""
import os, sys
import pyfbsdk as _pyfbsdk
if not hasattr(_pyfbsdk, "FBApplication"):
    import pyfbstandalone
    pyfbstandalone.initialize()
from pyfbsdk import *

src = os.path.abspath(sys.argv[1])
dst = os.path.abspath(sys.argv[2])

app = FBApplication()
app.FileNew()
app.FileOpen(src)

opt = FBFbxOptions(False)   # False = 保存选项
opt.UseASCIIFormat = False
saved = app.FileSave(dst, opt)
print("saved binary:", saved, dst, os.path.getsize(dst) if os.path.isfile(dst) else -1)
