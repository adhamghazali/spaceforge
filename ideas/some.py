import os

error_tars=['00060.tar',
'00153.tar',
'00161.tar',
'00193.tar',
'00208.tar',
'00224.tar',
'00232.tar',
'00242.tar',
'00251.tar',
'00269.tar',
'00323.tar',
'00365.tar',
'00416.tar',
'00514.tar',
'00542.tar',
'00595.tar',
'00687.tar',
'00688.tar',
'00696.tar',
'00757.tar',
'00765.tar',
'00766.tar',
'00767.tar',
'00839.tar',
'00884.tar',
'00938.tar',
'00944.tar',
'01008.tar',
'01014.tar',
'01016.tar',
'00279.tar',
'00381.tar',
'00384.tar',
'00549.tar',
'00597.tar',
'00700.tar',
'00793.tar',
'00794.tar',
'00898.tar',
'00899.tar',
'00901.tar',
'00005.tar',
'00299.tar',
'00304.tar',
'00312.tar',
'00343.tar',
'00349.tar',
'00402.tar',
'00412.tar',
'00435.tar',
'00444.tar',
'00486.tar',
'00487.tar',
'00491.tar',
'00499.tar',
'00552.tar',
'00556.tar',
'00559.tar',
'00573.tar',
'00581.tar',
'00613.tar',
'00677.tar',
'00712.tar',
'00719.tar',
'00722.tar',
'00750.tar',
'00810.tar',
'00812.tar',
'00825.tar',
'00852.tar',
'00883.tar',
'00902.tar',
'01049.tar',
'00961.tar',
'01061.tar',
'01086.tar',
'01095.tar',
'01096.tar',
'01134.tar',
'01142.tar',
'01162.tar',
'01163.tar',
'01166.tar',
'01192.tar',
'01229.tar'
]

print(error_tars)

directory='/datadrive4T/cc12m_w_embeds/'
for tar in error_tars:
    filename=directory+tar
    command='sudo rm '+filename
    print(command)
    os.system(command)