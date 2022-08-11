import tarfile

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


def test(filename, testfolder):

    temp_tar = tarfile.open(filename)
    try:
        temp_tar.extractall(testfolder) # specify which folder to extract to
        temp_tar.close()
        return True

    except:
        return False




import os
test_folder='/datadrive/cc2m/cc12m_w_embeds/'
files=listdir_fullpath(test_folder)
test_temp_folder='/datadrive/cc2m/temp_test'

if not os.path.exists(test_temp_folder):
  os.makedirs(test_temp_folder)


for filename in files:
    if test(filename, test_folder):
        continue
    else:
        print('Error in ', filename)


