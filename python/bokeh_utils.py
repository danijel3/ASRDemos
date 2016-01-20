from bokeh.plotting import figure, show, output_notebook
from bokeh.models import ColumnDataSource
import numpy as np

plots_initialized=False

class LinesPlot:

	color=['#ff7777','#117711','#7777ff','#ffff77','#ff77ff','#77ffff','#000000']

	def __init__(self,title,lines,width=900,height=300):
		
		global plots_initialized
		if not plots_initialized:
			plots_initialized=True
			output_notebook()

		self.P=figure(title=title)
		self.P.plot_width=width
		self.P.plot_height=height

		self.src=[]
		i=0
		for line in lines:
			s=ColumnDataSource(name=line)
			self.src.append(s)
			self.P.line([],[],source=s,legend=line,line_color=self.color[i])
			i+=1

		show(self.P)

	def add(self,idx,x,y):
		s=self.src[idx]

		if isinstance(x,(int,float,long)):
			s.data['x'].append(x)
			s.data['y'].append(y)
		else:
			s.data['x'].extend(x)
			s.data['y'].extend(y)

		s.push_notebook()

	def reset(self,idx):
		s=self.src[idx]
		s.data['x']=[]
		s.data['y']=[]
		s.push_notebook()

	def resetAll(self):
		for i in range(len(self.src)):
			self.reset(i)
