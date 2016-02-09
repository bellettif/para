
import sys
import fileinput
import re
import os
from pprint import pprint


def test_block_range(filename, max_block_size):
   block_sizes = [1] + range(16, max_block_size, 16)
   performance = {}

   for b in block_sizes:
      performance[b] = test_block_size(filename, b)

   return performance
   

def test_block_size(filename, block_size):
   inputfile = fileinput.input(filename, inplace = 1)
   for i, line in enumerate(inputfile):
      if i == 1:
         sys.stdout.write(re.sub(r'BLOCK_SIZE \d+',
                                 'BLOCK_SIZE ' + str(block_size), line))
      else:
         sys.stdout.write(line)
   inputfile.close()

   os.system('make')
   tmp = os.popen('./benchmark-auto_opt').read()

   return float(re.search('Average percentage of Peak = (\d+.\d+)', tmp).group(1))


if __name__ == "__main__":

   #Test many block sizes and report all of their performance
   #performances sorted from best to worst.
   max_block_size = 256
   results = test_block_range('dgemm-auto_opt.c', max_block_size)
   for block_size in sorted(results, key=results.get, reverse=True):
      print(' '.join(['Block Size: ', str(block_size),
                      '\t % Peak: ', str(results[block_size])]))
   
