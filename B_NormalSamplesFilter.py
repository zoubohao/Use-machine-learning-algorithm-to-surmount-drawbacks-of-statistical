import os , shutil
import re


if __name__ == "__main__":
    file_Dic = "D:\ThesisDataDownLoad\StomachCancer\SomaticMutation"
    normal_Dic = "D:\ThesisDataDownLoad\\NormalSample\SomaticMutation"
    if os.path.exists(normal_Dic) is False:
        os.mkdir(normal_Dic)
    for root , dics , files in os.walk(file_Dic):
        for file in files:
            fileSplit = file.split("_")
            ID = fileSplit[0]
            IDSplit = ID.split("-")
            if int(IDSplit[-1]) >= 11:
                shutil.move(os.path.join(root,file) , os.path.join(normal_Dic,file))






