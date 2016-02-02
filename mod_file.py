
import sys
import fileinput
import re

inputfile = fileinput.input('sample_file.txt', inplace = 1)

for i, line in enumerate(inputfile):
   if i == 1:
      sys.stdout.write(re.sub(r'BLOCK_SIZE \d+', 'BLOCK_SIZE 33', line))
   else:
      sys.stdout.write(line)


tmp = os.popen('./benchmark-blas').read()

float(re.search('Average percentage of Peak = (\d+.\d+)', tmp).group(1))