import sys
import h5py
import numpy as np
from random import shuffle

class Corpus:

    def __init__(self,filename,utts_loaded=None,load_normalized=False,merge_utts=False):        

        self.filename=filename
        self.utts_loaded=utts_loaded
        self.load_normalized=load_normalized        
        self.merge_utts=merge_utts

        self.h5f=h5py.File(filename,'r')
        self.utts=[]
        for utt in self.h5f.keys():
            self.utts.append(utt)

        self.reset_utts_loaded(utts_loaded)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def reset_utts_loaded(self,utts_loaded):

        if not utts_loaded:
            utts_loaded=len(self.utts)

        un=len(self.utts)

        self.r=range(0,un,utts_loaded)
        self.n=len(self.r)
        self.r.append(un)                

        self.reset()


    def split(self,ratio):
        a=Corpus(filename=self.filename,utts_loaded=self.utts_loaded,
            load_normalized=self.load_normalized,merge_utts=self.merge_utts)
        b=Corpus(filename=self.filename,utts_loaded=self.utts_loaded,
            load_normalized=self.load_normalized,merge_utts=self.merge_utts)

        un_r=int(len(self.utts)*ratio)

        a.utts=self.utts[:un_r]
        b.utts=self.utts[un_r:]

        a.reset_utts_loaded(self.utts_loaded)
        b.reset_utts_loaded(self.utts_loaded)

        return (a,b)


    def __iter__(self):
        for c in range(self.n):
            yield self.get(slice(self.r[c],self.r[c+1]))


    def reset(self):
        shuffle(self.utts)

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

        if self.merge_utts:
            inputs=np.vstack(inputs)
            outputs=np.concatenate(outputs)
        else:
            inputs=np.array(inputs)
            outputs=np.array(outputs)

        return (inputs,outputs)

    def getMaxLen(self):
        m=0
        for u in self.h5f:    
            if m<self.h5f[u+'/out'].size:
                m=self.h5f[u+'/out'].size
        return m

class Report:
    def __init__(self):
        self.loss=[]
        self.acc=[]
        self.samp_num=[]
    
    def add(self,loss,acc=None,samp_num=None):
        self.loss.append(loss)
        if acc:
            self.acc.append(acc)
        if samp_num:
            self.samp_num.append(samp_num)
        
    def getMeanLoss(self):
        return np.mean(self.loss)
    
    def getAveAcc(self):
        if len(self.acc)==0:
            return 0
        else:
            return np.average(self.acc,weights=self.samp_num)
    
    def printout(self,title):
        print '{} loss: {}'.format(title,self.getMeanLoss())
        if len(self.acc)>0:
            print '{} PER: {:%}'.format(title,1-self.getAveAcc())     
        sys.stdout.flush()

class History:
    def __init__(self, name):
        self.name=name
        self.r=Report()
        self.loss=[]
        self.acc=[]

    def log(self):
        self.r.printout(self.name)
        self.loss.append(self.r.getMeanLoss())
        self.acc.append(self.r.getAveAcc())
        self.r=Report()

