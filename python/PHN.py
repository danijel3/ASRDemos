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

class PHN:
	
	def __init__(self):
		self.segments = []		
		
	def __str__(self):
		return str(self.segments)
	
	def __repr__(self):
		return self.__str__()

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

	
	re_line = re.compile("^([0-9]+) ([0-9]+) (.+)$")
	
	def parseLine(self, line):
		
		m = self.re_line.match(line)
		
		assert m is not None
		
		return int(m.group(1)),int(m.group(2)),m.group(3)
	
	def load(self, filename):
		
		with open(filename, 'r') as f:
			for line in f:
				xmin,xmax,text=self.parseLine(line)
				self.segments.append(Segment(xmin,xmax,text))
						
	def save(self, filename):
				
		with open(filename, 'w') as f:
			
			for seg in self.segments:				
				
				f.write('{} {} {}\n'.format(seg.xmin,seg.xmax,seg.text))