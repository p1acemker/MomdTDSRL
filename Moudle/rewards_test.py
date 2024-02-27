# 两个def
# 外层def 负责受体文件的遍历，需要的参数pdb文件的路径，配体的路径，所给最大的资源，结果文件的主路径
# 内层def 生成 num个.in文件
# 其中 padbpath 是receptorlist，需要做一个校验，看当前的receptor下有木有.in文件，还需要截取到其父亲文件夹
# 然后创建他们的父文件夹用来存放新生成的.in文件


import os
import shutil
import time


def receptor_qsub(qsub_path, receptorlistpath):
    start_time = time.time()
    qsublist = qsub_path.rpartition('/')[0] + "/qsub"
    if os.path.exists(qsublist):
        shutil.rmtree(qsublist)
        os.mkdir(qsublist)
    else:
        os.mkdir(qsublist)

    receptor_open = open(receptorlistpath, 'r', encoding='utf-8')
    receptor_list = receptor_open.read().splitlines()
    receptor_list_lines = len(receptor_list)
    for line in range(0, receptor_list_lines):
        getline = receptor_list[line]
        pro_pdb_name = getline.rpartition('/')[0].rpartition('/')[2]
        file_qsub = open(qsublist + '/' + pro_pdb_name + '.pbs', 'a', encoding='utf-8')
        file_qsub.write('#!/bin/bash' + '\n')
        file_qsub.write('#PBS -N ledock-' + pro_pdb_name + '\n')
        file_qsub.write('#PBS -l nodes=100:ppn=1' + '\n')
        file_qsub.write('#PBS -l walltime=999:00:00' + '\n')
        file_qsub.write('#PBS -q q_csywz' + '\n')
        file_qsub.write('#PBS -V' + '\n')
        file_qsub.write('#PBS -S /bin/bash' + '\n')
        file_qsub.write('#PBS -o /home/csywz/deep/ledock/logs/ledock-' + pro_pdb_name + '.log' + '\n')
        file_qsub.write('\n')
        file_qsub.write('date' + '\n')
        file_qsub.write('mpiexec -n 100 python /home/csywz/deep/Modules/mpi_ledock.py ' + getline + '\n')
        file_qsub.write('date' + '\n')
        file_qsub.close()
    os.remove(receptorlistpath)
    print("生成脚本所使用时间：%s" % (time.time() - start_time))
    return qsublist


def Split_file_pdb(num, pdbpath_list, ligandpath):
    # 将受体文件生成num个.in文件
    # 同时建成相应的结果文件夹
    # pdbpath_list中的每一条都是 ledock.in 文件的绝对路径
    start_time = time.time()
    print("The program is running Split_file_pdb")
    try:
        receptor = open(pdbpath_list, 'r', encoding='utf-8')
        receptor = receptor.read()
        line_list = receptor.splitlines()
        lines_num = len(line_list)
        for i in range(0, lines_num):
            getline = line_list[i]
            n = getline.rfind('/')
            m = getline.rfind('/', 0, n)
            receptor_file = getline[m + 1: n]  # 得到受体文件的文件夹的名字,同时也是受体文件的名字
            ledock_file_open = open(getline, 'r', encoding='utf-8')
            ledock_list = ledock_file_open.read().splitlines()
            receptor_ledock = getline[0:n] + '/' + receptor_file + '_ledockin'
            receptor_pdb = getline[0:n] + '/' + receptor_file + '_pro.pdb'
            if os.path.exists(receptor_ledock):
                shutil.rmtree(receptor_ledock)
                os.mkdir(receptor_ledock)
            else:
                os.mkdir(receptor_ledock)
            for j in range(0, num):
                receptor_ledock_new = receptor_ledock + '/' + receptor_file + '_' + 'ligandlist' + str(j) + '.in'
                receptor_ledock_file = open(receptor_ledock_new, 'w', encoding='utf-8')

                for line in range(0, 17):
                    if line == 1:
                        receptor_ledock_file.write(receptor_pdb + '\n')
                    elif line == 15:
                        receptor_ledock_file.write(ligandpath + '/ligandlist' + str(j) + '\n')
                    else:
                        receptor_ledock_file.write(ledock_list[line] + '\n')

            file_result = getline[0:n] + '/' + receptor_file + '_result'
            if os.path.exists(file_result):
                shutil.rmtree(file_result)
                os.mkdir(file_result)
            else:
                os.mkdir(file_result)

            ledock_file_open.close()
        receptor_ledock_file.close()
        print("生成受体配置文件所使用时间：%s" % (time.time() - start_time))
    except:
        print("Split_file_pdb is error ")


def Split_file_ligand(num, file_ligand):
    start_time = time.time()
    print("The program is running Split_file_ligand")
    try:
        ligandPath = open(file_ligand, 'r', encoding='utf-8')
        temp = ligandPath.read()
        line_list = temp.splitlines()
        lines_num = len(line_list)

        ligand_count = int(lines_num / num)
        ligand_count_last = int(lines_num % num)
        file_ligandPath = file_ligand.rpartition('/')[0] + '/ligandlist'

        if os.path.exists(file_ligandPath):
            shutil.rmtree(file_ligandPath)
            os.mkdir(file_ligandPath)
        else:
            os.mkdir(file_ligandPath)

        for i in range(0, num):
            file_ligandpath = file_ligandPath + '/ligandlist' + str(i)  # 创建新的ligandlist
            file = open(file_ligandpath, 'a', encoding='utf-8')
            for j in range(i * ligand_count, (i + 1) * ligand_count):
                getline = line_list[j]
                file.write(getline + '\n')
            file.close()

        # TODO 余数放在最后一个文件
        if (ligand_count_last != 0):
            file = open(file_ligandpath, 'a', encoding='utf-8')
            start_line = num * ligand_count
            end_line = lines_num
            for j in range(start_line, end_line):
                getline = line_list[j]
                file.write(getline + '\n')
            file.close()

        ligandPath.close()
        print("切割配体文件所使用时间：%s" % (time.time() - start_time))
        return file_ligandPath
    except:
        print("The ligand cannot open")


def get_filelist(dir):
    Filelist = []
    for home, dirs, files in os.walk(dir):
        for filename in files:
            # 文件名列表，包含完整路径
            Filelist.append(os.path.join(home, filename))
            # # 文件名列表，只包含文件名
            # Filelist.append( filename)

    return Filelist


if __name__ == "__main__":
    try:
        receptorpath = "/home/csywz/deep/ledock/Mpro/Mpro"
        file_ligand = "/home/csywz/deep/ledock/ligand02"
        resources = 100 * 28
        num = int(resources / 28)

        # 切割配体文件列表
        file_ligand_new_path = Split_file_ligand(num, file_ligand)

        pdblist = file_ligand.rpartition('/')[0] + "/pdblist"
        receptorlistpath = file_ligand.rpartition('/')[0] + "/receptorlistpath"
        pdblist_open = open(pdblist, 'w', encoding='utf-8')
        receptor_open = open(receptorlistpath, 'w', encoding='utf-8')

        filelist = get_filelist(receptorpath)
        for file in filelist:
            if file.endswith('_ledock.in'):
                pdblist_open.write(file + '\n')
                receptor_open.write(file.replace('_ledock.in', '_pro.pdb') + '\n')

        pdblist_open.close()
        receptor_open.close()

        Split_file_pdb(num, pdblist, file_ligand_new_path)
        os.remove(pdblist)

        # 生成qsub脚本
        qsublist = receptor_qsub(file_ligand, receptorlistpath)
        qsubfilelist = get_filelist(qsublist)

        # 提交任务
        for qsub in qsubfilelist:
            result = os.system("qsub " + qsub)
            print(result)
    except:
        print("出现错误！")
