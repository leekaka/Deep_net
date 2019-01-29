#coding：utf8
import os

def rename():
     path = "E:\\path\\"
     filelist = os.listdir(path)

     for files in filelist:
          Olddir = os.path.join(path,files)
          if os.path.isdir(Olddir):
               continue
          filename = os.path.splitext(files)[0]
          filetype = os.path.splitext(files)[1]
          filename = str(filename)
		  
          #forename = filename.split("C")[0]
          #realname = filename.split("C")[1]
          #num = int(realname)
          #num = str(num - 31)
          #print(num.zfill(5))
          #namelist = "DSC" + num.zfill(5)
		  
		  namelist = "new_name"              # 根据需求改名字
          Newdir = os.path.join(path,namelist+filetype);
          os.rename(Olddir,Newdir)#重命名  
rename()
