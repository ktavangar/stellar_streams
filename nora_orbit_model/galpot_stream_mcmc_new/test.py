from contextlib import closing
from multiprocessing import Pool

with closing(Pool(processes=2)) as pool:
    print('hello')
