"""Parser used to get the data, create 100 word references
and return samples. 

CODE IS AMMENDED FROM: 
https://bitbucket.org/tanya14109/cqasumm/src/master/

Returns:
	[CQA_DOCUMENT]
"""
import sys 
sys.path.append(".")
from parser import parseDataSet
from config import Config
from pointerDataset import buildPointerGeneratorDataset

def main():
    	
	config = Config()
	samples = parseDataSet(config)
	buildPointerGeneratorDataset(samples)	

if __name__ == '__main__':
    main()