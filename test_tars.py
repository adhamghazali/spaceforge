import tarfile
import os
import sys 

test_folder='/datadrive4T/cc12m_w_embeds/'

temp='/datadrive4T/temp_test_tars'


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


def test(filename, temp):

    temp_tar = tarfile.open(filename)
    try:
        temp_tar.extractall(temp) # specify which folder to extract to
        temp_tar.close()
        return True

    except:
        return False


if not os.path.exists(temp):
  os.makedirs(temp)


command='sudo rm -r '+temp
cc=0
file_path = 'tars.txt'
sys.stdout = open(file_path, "w")

files=listdir_fullpath(test_folder)
for filename in files:
    cc+=1
    if cc%10==0:
        os.system(command)
        os.makedirs(temp)

    print(filename)
    if test(filename, temp):
        continue
    else:
        print('Error in ', filename)


os.system(command)


