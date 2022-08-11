import tarfile
import os

test_folder='/datadrive/cc2m/cc12m_w_embeds/'

temp='/datadrive/cc2m/temp_test'


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

files=listdir_fullpath(test_folder)
for filename in files:
    print(filename)
    if test(filename, temp):
        continue
    else:
        print('Error in ', filename)

commnad='sudo rm -r '+temp
os.system(commnad)


