import os

def pick_un_transed(ori_path,transed_path):
    ori_dict={}
    un_transed_list=[]

    for paths,dirs,files in os.walk(ori_path):
        for file in files:
            ori_dict[file]=0
    for paths, dirs, files in os.walk(transed_path):
        for file in files:
            ori_dict[file] = 1
    for key,value in ori_dict:
        if value==0:
            un_transed_list.append(key)
    return un_transed_list

mol2_list=os.listdir("mol2")

for i in mol2_list:
    open('mol2/list.txt','a').write(i+'\n')
