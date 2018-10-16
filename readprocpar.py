#!/usr/bin/python3

from argparse import ArgumentParser

# hlatkydavid@gmail.com 2018.07.11

class procparReader():
	"""
	class to read .procpar files and format it into a dictionary {parameter : value}
	
	INPUT: 'procpar_file'
	
	METHODS:

	read() : reads file and makes dictionary

	"""
	def __init__(self, procpar_file):
			self.ppfile = str(procpar_file)
				
	def read_deprecated(self):

	# ----------------------- defining helper functions----------------------------
		def represents_int(s):
			try: 
				int(s)
				return True
			except ValueError:
				return False		

		def format(ppdict):

			temp_dict = ppdict
			for k in temp_dict.keys():
				if k.startswith('"'):
					k = k.replace('"','')
				if temp_dict[k].startswith('"'):
					val = temp_dict[k].replace('"','')
				else:
					val = temp_dict[k]
				ppdict[k] = val
			return ppdict

	# -----------------read main ----------------------------------------------
		with open(self.ppfile,'r') as openpp:
		# TODO reading parameters with multiple strings
			names = []
			vals = []
			self.ppdict = {}
			k = 0; temp_num = -1
			for num, line in enumerate(openpp.readlines()):
				line = line.replace('\n','')
				if (represents_int(line[0]) == False and line[0] != '"'):
					temp_num = num
					name = line.split(' ')[0]
					names.append(name)

				if (num == temp_num+1):
					val = line.split(' ')[1]
					vals.append(val)
					self.ppdict[names[k]] = vals[k]
					k = k+1

		self.ppdict = format(self.ppdict) # additional formatting, might not be needed
		return self.ppdict

	def read(self):
		"""
		THE definitive read method, use this and not the other garbage

		"""

		with open(self.ppfile,'r') as openpp:
			
			name = []
			value = []
			subtype = []
			basictype = []
			maxvalue = []
			minvalue = []
			
			line_id = 0 # possibilities (as of manual): 0 :'line1', 1: 'line2', 2: 'last'
		
			current_name = 0
	
			for line_num, line in enumerate(openpp.readlines()):

				if line_id == 0:  # parameter name line
					
					fields = line.rsplit('\n')[0].rsplit(' ')
					current_name = fields[0]
					name.append(fields[0])
					subtype.append(int(fields[1]))
					basictype.append(int(fields[2]))

						
					current_basictype = int(fields[2])  # can be 0 (undefined), 1 (real), 2 (string)

					val_list = []  # if parameter value is a multitude of strings
					line_id = 1
									
				elif line_id == 1:  # parameter value line

					fields = line.rsplit('\n')[0].rsplit(' ')

					if current_basictype == 1:  # values are real, and all are on this line

						val = [fields[i] for i in range(1,len(fields)-1)]
						if len(val) == 1:  # don't make list if there is only one value
							val = val[0]
						value.append(val)
						line_id = 2

					elif current_basictype == 2 and int(fields[0]) == 1: # values are strings
						
						value.append(str(fields[1].rsplit('"')[1]))
						line_id = 2
					
					elif current_basictype == 2 and int(fields[0]) > 1: # multiple string values
						
						val_list.append(fields[1])
						remaining_values = int(fields[0])-1
						line_id = 3
					
				elif line_id == 2:

					line_id = 0
			
				elif line_id == 3:  # if values are on multiple lines
					if remaining_values > 1:  # count backwards from remaining values
						val_list.append(fields[0])
						remaining_values = remaining_values - 1
					elif remaining_values == 1:
						val_list.append(fields[0])
						remaining_values = remaining_values - 1
						value.append(val_list)
						line_id = 2
					else:
						print('something is off')

				else:
					print('incorrect line_id')


		return dict(zip(name, value))
# utility for testing
if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument('inputpp')
	args = parser.parse_args()

	#ppfile = str(args.inDir)+'/procpar'
	print(args.inputpp)
	#print(ppfile)
	ppr = procparReader(args.inputpp)
	#ppdict = ppr.read_on_steroids()
	ppdict = ppr.read()
	#print(ppdict)

	print(ppdict['orient'])
	#print(ppdict.keys())
