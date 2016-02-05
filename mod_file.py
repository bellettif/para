
import sys
import fileinput
import re
from pprint import pprint


def test_block_range(inputfile, max_block_size):
   block_sizes = range(1, max_block_size)
   performance = {}

   for b in block_sizes:
      performance[b] = test_block_size(b)

   return performance

   

def test_block_size(inputfile, block_size):
   for i, line in enumerate(inputfile):
      if i == 1:
         sys.stdout.write(re.sub(r'BLOCK_SIZE \d+',
                                 'BLOCK_SIZE ' + str(block_size), line))
      else:
         sys.stdout.write(line)


   tmp = os.popen('./benchmark-blas').read()

   return float(re.search('Average percentage of Peak = (\d+.\d+)', tmp).group(1))


if __name__ == "__main__":

   #Test many block sizes and report all of their performance
   #performances sorted from best to worst. 
   inputfile = fileinput.input('sample_file.txt', inplace = 1)
   max_block_size = 256
   results = test_block_range(inputfile, max_block_size)
   for block_size in sorted(results, key=results.get, reverse=true)::
      print(' '.join(['Block Size: ', str(block_size),
                      '\t % Peak: ', str(results[block_size])]))
   
