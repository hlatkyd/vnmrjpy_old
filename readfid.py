#!/usr/bin/python3

import os
import numpy as np
from argparse import ArgumentParser

# hlatkydavid@gmail.com 2018.07.11

class fidReader():
	"""
	INPUT : fid - .fid file
			procpar - .procpar file, (both should be in the same directory)
	METHODS:
			__init__(fid,ppfile) : reads in binary fid separates fid header and
					data, and reads procpar

			print_header(): fid prints header

			read() : separates binary data to blocks according to 'nblocks' in 
					 header. nblocks = receivers * (repetitions + reference scans)
					 makes a list of block binary data

					 return DATA = np.ndarray(blocks,traces*np)

	"""
#--------------------------------------------------------------------------------------------------
#
# 										READING IN BINARY DATA
#
#--------------------------------------------------------------------------------------------------
	"""
	--------------------------------------------------------------------------
	struct datafilehead
	Used at the beginning of each data file (fid's, spectra, 2D) 

	   int     nblocks;      /* number of blocks in file
	   int     ntraces;      /* number of traces per block	
	   int     np;           /* number of elements per trace	
	   int     ebytes;       /* number of bytes per element			
	   int     tbytes;       /* number of bytes per trace		
	   int     bbytes;       /* number of bytes per block		
	   short   vers_id;      /* software version and file_id status bits
	   short   status;       /* status of whole file	
	   int	   nbheaders;	 /* number of block headers			

	struct datablockhead
 	Each file block contains the following header 

	   short   scale;	/* scaling factor        
	   short   status;	/* status of data in block      
	   short   index;	/* block index              
	   short   mode;	/* mode of data in block	 
	   int	   ctcount;	/* ct value for FID		
	   float   lpval;	/* F2 left phase in phasefile    
	   float   rpval;	/* F2 right phase in phasefile
	   float   lvl;		/* F2 level drift correction       
	   float   tlt;		/* F2 tilt drift correction      
	"""

	def __init__(self,fid,procpar=None):
			
		if os.path.isdir(fid):
			fid = str(fid)+'/fid'
			procpar = str(fid)+'/procpar'
		else:			
			pass

		def decode_header():

			hkey_list = ['nblocks','ntraces','np','ebytes','tbytes','bbytes',\
					 'vers_id','status','nbheaders']
			hval_list = []
			h_dict = {}
			if len(self.bheader) != 32:
				print('incorrect fid header data: not 32 bytes')
				return -1
			else: 
				hval_list.append(int.from_bytes(self.bheader[0:4],byteorder='big'))
				hval_list.append(int.from_bytes(self.bheader[4:8],byteorder='big'))
				hval_list.append(int.from_bytes(self.bheader[8:12],byteorder='big'))
				hval_list.append(int.from_bytes(self.bheader[12:16],byteorder='big'))
				hval_list.append(int.from_bytes(self.bheader[16:20],byteorder='big'))
				hval_list.append(int.from_bytes(self.bheader[20:24],byteorder='big'))
				hval_list.append(int.from_bytes(self.bheader[24:26],byteorder='big'))
				hval_list.append(int.from_bytes(self.bheader[26:28],byteorder='big'))
				hval_list.append(int.from_bytes(self.bheader[28:],byteorder='big'))

				for num, i in enumerate(hkey_list):
					h_dict[i] = hval_list[num]
			
				self.header_dict = h_dict

				def decode_status(): #TODO
					pass

		# data should be 32 bytes
		with open(fid,'rb') as openFid:
			binary_data = openFid.read() 
			self.bheader = bytearray(binary_data[:32])
			self.bdata = binary_data[32:]
		decode_header()

	def read(self):

		def chunks(l, n):
			for i in range(0, len(l), n):
				yield l[i:i + n]
		#chunk_size = int(len(self.bdata) / self.header_dict['nblocks'])
		chunk_size = int(self.header_dict['bbytes'])
		block_list = list(chunks(self.bdata,chunk_size))

		if self.header_dict['bbytes'] == len(block_list[0]):
			print('blocksize check OK')

		def decode_blockhead(blockheader):
			bh_dict = {}
			bhk_list = ['scale','status','index','mode','ctcount','lpval'\
						'rpval','lvl','tlt']
			bhv_list = []
			bhv_list.append(int.from_bytes(blockheader[0:2],byteorder='big'))
			bhv_list.append(int.from_bytes(blockheader[2:4],byteorder='big'))
			bhv_list.append(int.from_bytes(blockheader[4:6],byteorder='big'))
			bhv_list.append(int.from_bytes(blockheader[6:8],byteorder='big'))
			bhv_list.append(int.from_bytes(blockheader[8:12],byteorder='big'))
			bhv_list.append(int.from_bytes(blockheader[12:16],byteorder='big'))
			bhv_list.append(int.from_bytes(blockheader[16:20],byteorder='big'))
			bhv_list.append(int.from_bytes(blockheader[20:24],byteorder='big'))
			bhv_list.append(int.from_bytes(blockheader[24:28],byteorder='big'))

			for num, i in enumerate(bhk_list):
				bh_dict[i] = bhv_list[num]		
			self.blockhead_dict = bh_dict

		#------------main iteration through bytearray -----------------

		dim = (int(self.header_dict['nblocks']),int((self.header_dict['ntraces'])*int(self.header_dict['np'])))
		DATA = np.empty(dim)
		print(DATA.shape)
		if self.header_dict['ebytes'] == 4:
			dt = '>f'
		if self.header_dict['ebytes'] == 2:
			dt = '>i2'
		print('blocknum: ' +str(len(block_list)))
		for k,block in enumerate(block_list): # for each block
			block_header = block[:28] # separate block header
			block_data = block[28:]
			DATA[k,:] = np.frombuffer(bytearray(block_data),dt)

		return DATA, self.header_dict

	def print_header(self):

		print('-----------printing header--------------------------')
		for i in sorted(self.header_dict.keys()):
			print(str(i)+' = '+str(self.header_dict[i]))

#--------------------------------------------------------------------------------------------------
#utility for testing
def main(infid):

	fid = fidReader(infid)
	fid.print_header()
	data = fid.read()
	
if __name__ == '__main__':

	parser = ArgumentParser()
	parser.add_argument('fid',help='input .fid file')
	args = parser.parse_args()

	if os.path.isdir(args.fid):
		fid = str(args.fid)+'/fid'
		pp = str(args.fid)+'/procpar'
	else:
		fid = args.fid
	main(fid)	

