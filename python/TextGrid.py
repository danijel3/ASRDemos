# -*- coding: utf-8 -*-
import re
import numpy

class Segment:
	def __init__(self, xmin=0, xmax=0, text=''):
		self.__dict__.update(locals())
		
	def __str__(self):
		return str('"' + self.text + '" : ' + str(self.xmin) + ' -- ' + str(self.xmax))
	
	def __repr__(self):
		return self.__str__()

class Tier:
	def __init__(self):
		self.xmin = 0
		self.xmax = 0
		self.segments_size = 0
		self.segments = []
		self.name = ''
		
	def fromSequence(self, seq, score, seqdict, timestep):
		
		self.segments = []
		on = seq[0]
		ot = 0
		for i, n in enumerate(seq):
			if(n != on):
				p = numpy.mean(score[ot:i, on])
				text = '{} <{:.2%}>'.format(seqdict[on], p)
				self.segments.append(Segment(ot * timestep, i * timestep, text))
				on = n
				ot = i
				
		i = seq.size
		p = numpy.mean(score[ot:i, on])
		text = '{} <{:.2%}>'.format(seqdict[on], p)
		self.segments.append(Segment(ot * timestep, i * timestep, text))
		self.xmin = 0
		self.xmax = i * timestep
		self.segments_size = len(self.segments)
		
		
	def fromSequence(self, seq, timestep):
		
		self.segments = []
		on = seq[0]
		ot = 0
		for i, n in enumerate(seq):
			if(n != on):				
				text = str(on)
				self.segments.append(Segment(ot * timestep, i * timestep, text))
				on = n
				ot = i
				
		i = seq.size
		text = str(on)
		self.segments.append(Segment(ot * timestep, i * timestep, text))
		self.xmin = 0
		self.xmax = i * timestep
		self.segments_size = len(self.segments)
			
	def getCode(self, beg, end, nul_val='!'):
		for s in self.segments:
			if(end < s.xmin):
				continue
			if(beg > s.xmax):
				continue
			
			if(end <= s.xmax and beg >= s.xmin):
				return s.text
			
			
			if(beg <= s.xmin):
				if(end - s.xmin >= s.xmin - beg):
					return s.text
			
			if(end >= s.xmax):
				if(s.xmax - beg >= end - s.xmax):
					return s.text
			
		return nul_val
	
	
	def toSequence(self, nSamples, win_shift, win_size, code_mapping=None, nul_val=-1):
		codes = []
		for s in range(nSamples):
			beg = s * win_shift
			end = beg + win_size
			c = self.getCode(beg, end).strip()
			if code_mapping != None:
				if c in code_mapping:
					codes.append(code_mapping[c])
				else:
					codes.append(nul_val)
			else:
				codes.append(c)
				
		return codes
	
	def __str__(self):
		return '[' + self.name + '] {' + str(self.segments) + '}'
	
	def __repr__(self):
		return self.__str__()

class Line:
	def __init__(self, linetype='', name='', value=None):
		self.__dict__.update(locals())

class TextGrid:
	
	def __init__(self):
		self.xmin = 0
		self.xmax = 0
		self.tiers_size = 0
		self.tiers = []
		
	def __str__(self):
		return str(self.tiers)
	
	def __repr__(self):
		return self.__str__()

	
	re_empty = re.compile("^\s*$")
	keyvalstr = re.compile("\s*(\w+[ \w:]*\w)\s?=\s?\"([\w ]*)\"")
	keyvalstrpl = re.compile("\s*(\w+[ \w:]*\w)\s?=\s?\"([^\"]*)\"")
	keyvalnum = re.compile("\s*(\w+[ \w:]*\w)\s?=\s?([0-9\.]*)")
	keyvalbool = re.compile("\s*(\w+[ \w:]*\w)\s?\?\s?([\w<>]*)")
	re_array = re.compile("\s*(\w+[ \w]*\w)\s?\[([0-9]*)\]:")
	
	def parseLine(self, line):
		
		m = self.re_empty.match(line)
		if(m is not None):
			return Line('EMPTY')
		
		m = self.keyvalstr.match(line)
		if(m is not None):
			return Line('STR', m.group(1), m.group(2))
		
		m = self.keyvalstrpl.match(line)
		if(m is not None):
			return Line('STRPL', m.group(1), m.group(2))
		
		m = self.keyvalnum.match(line)
		if(m is not None):
			return Line('NUM', m.group(1), m.group(2))

		m = self.keyvalbool.match(line)
		if(m is not None):
			return Line('BOOL', m.group(1), m.group(2))

		m = self.re_array.match(line)
		if(m is not None):
			return Line('ARR', m.group(1), m.group(2))
		
		return Line('??')
	
	def load(self, filename):
		
		with open(filename, 'r') as f:
			
			while True:
				
				line = f.readline()				
				
				if(line == ''):
					return
				
				p = self.parseLine(line)
				
				if(p.linetype == 'EMPTY'):
					continue
				
				if(p.name == 'File type'):
					assert p.value == 'ooTextFile'
				
				if(p.name == 'Object class'):
					assert p.value == 'TextGrid'
				
				if(p.name == 'xmin'):
					self.xmin = float(p.value)
				
				if(p.name == 'xmax'):
					self.xmax = float(p.value)
					
				if(p.linetype == 'BOOL' and p.name == 'tiers'):
					if(p.value != '<exists>'):
						return
				
				if(p.name == 'size'):
					self.tiers_size = int(p.value)
				
				if(p.linetype == 'ARR' and p.name == 'item' and p.value == ''):
					
					for ti in range(self.tiers_size):						
						
						t = Tier()
						
						p = self.parseLine(f.readline())
						assert p.linetype == 'ARR' and p.name == 'item'
						if p.value != str(ti + 1):
							print "WARING: item number is wrong in tier: " + p.value + " should be " + str(ti + 1)
						p = self.parseLine(f.readline())
						assert p.name == 'class' and p.value == 'IntervalTier'
						
						p = self.parseLine(f.readline())
						assert p.name == 'name'
						t.name = p.value
						
						p = self.parseLine(f.readline())
						assert p.name == 'xmin'
						t.xmin = float(p.value)
						
						p = self.parseLine(f.readline())
						assert p.name == 'xmax'
						t.xmax = float(p.value)
						
						p = self.parseLine(f.readline())
						assert p.name == 'intervals: size'
						t.segments_size = int(p.value)
						
						for si in range(t.segments_size):
							
							s = Segment()
							
							p = self.parseLine(f.readline())
							assert p.linetype == 'ARR' and p.name == 'intervals' and p.value == str(si + 1)
							
							p = self.parseLine(f.readline())
							assert p.name == 'xmin'
							s.xmin = float(p.value)
							
							p = self.parseLine(f.readline())
							assert p.name == 'xmax'
							s.xmax = float(p.value)
							
							p = self.parseLine(f.readline())
							assert p.name == 'text'
							s.text = p.value							
							
							t.segments.append(s)
							
						self.tiers.append(t)
						
	def save(self, filename):
		
		
		for t in self.tiers:
			for s in t.segments:
				if(t.xmax < s.xmax):
					t.xmax = s.xmax
			if(self.xmax < t.xmax):
				self.xmax = t.xmax
		
		with open(filename, 'w') as f:
			
			f.write('File type ="ooTextFile"\n')
			f.write('Object class = "TextGrid"\n')
			f.write('\n')
			f.write('xmin = ' + str(self.xmin) + '\n')
			f.write('xmax = ' + str(self.xmax) + '\n')
			f.write('tiers? <exists>\n')
			f.write('size = ' + str(len(self.tiers)) + '\n')
			f.write('item []:\n')
			
			for t in range(len(self.tiers)):
				
				tier = self.tiers[t]
				
				f.write('\titem [' + str(t + 1) + ']:\n')
				f.write('\t\tclass = "IntervalTier"\n')
				f.write('\t\tname = "' + tier.name + '"\n')
				f.write('\t\txmin = ' + str(tier.xmin) + '\n')
				f.write('\t\txmax = ' + str(tier.xmax) + '\n')
				f.write('\t\tintervals: size = ' + str(len(tier.segments)) + '\n')
				
				for s in range(len(tier.segments)):
					
					seg = tier.segments[s]
					
					f.write('\t\tintervals [' + str(s + 1) + ']:\n')
					f.write('\t\t\txmin = ' + str(seg.xmin) + '\n')
					f.write('\t\t\txmax = ' + str(seg.xmax) + '\n')
					f.write('\t\t\ttext = "' + seg.text + '"\n')
