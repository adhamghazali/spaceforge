
import json
from glob import glob

def compute_stats(files, limit=None):
    stats_files = glob(files)
    #print(stats_files)
    

    n_total = 0
    n_successes = 0
    for stats_file in stats_files[0:limit] if limit is not None else stats_files:
        #print(stats_file)
        f = open(stats_file)
        stats = json.load(f)
        f.close()
        n_total += stats["count"]
        n_successes += stats["successes"]
        

    print(
       f"Files: [{stats_files[0]} ... {stats_files[limit] if limit is not None else stats_files[-1]}] \n"\
       f"Count: {n_total:,} \n"\
       f"Successes: {n_successes:,} \n"\
       f"Success rate: {(n_successes/n_total)*100:.2f}% \n"\
       f"Steps (bs 256): {n_successes/256:,.0f}")




#files=[
#        '/datadrive/cc2m/cc12m/00001_stats.json',
#        '/datadrive/cc2m/cc12m/00002_stats.json',
#        '/datadrive/cc2m/cc12m/00003_stats.json',
#        '/datadrive/cc2m/cc12m/00004_stats.json',
#        '/datadrive/cc2m/cc12m/00005_stats.json'
#       ]
#compute_stats(files)
compute_stats("/datadrive/cc2m/cc12m/*_stats.json" )

