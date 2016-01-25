import h5py
import numpy as np
from random import shuffle

class Corpus:

    def __init__(self,filename,utts_loaded=None,load_normalized=False):        
        self.h5f=h5py.File(filename,'r')
        self.utts=[]
        for utt in self.h5f.keys():
            self.utts.append(utt)

        if not utts_loaded:
            utts_loaded=len(self.utts)

        un=len(self.utts)

        self.r=range(0,un,utts_loaded)
        self.n=len(self.r)
        self.r.append(un)        
        shuffle(self.utts)
        self.c=0

        self.load_normalized=load_normalized

    def __iter__(self):
        return self

    def next(self):
        if self.c>=self.n:
            raise StopIteration
        else:   
            self.c+=1
            return get(slice(self.r[self.c],self.r[self.c+1]))


    def reset(self):
        shuffle(self.utts)
        self.c=0

    def close(self):
        self.h5f.close()

    def get(self,s=None):

        if self.load_normalized:
            in_name='norm'
        else:
            in_name='in'

        if s==None:
            s=slice(None,None)

        inputs=[]
        outputs=[]        
        for utt in self.utts[s]:
            g=self.h5f[utt]
            inputs.append(g[in_name][()])
            outputs.append(g['out'][()])

        return (inputs,outputs)
