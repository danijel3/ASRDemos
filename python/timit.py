import sys
import numpy as np
from scipy.io.wavfile import read
import os
import h5py
from tqdm import *

sys.path.append('../PyHTK/python')

from HTKFeat import MFCC_HTK


class Segment:
    def __init__(self):
        self.beg=0
        self.end=0
        self.text=''
    def __str__(self):
        return '({},{}) "{}" '.format(self.beg,self.end,self.text)

def load_mlf(path):    
    with open(path,'r') as F:
        
        assert F.readline()[:-1] == '#!MLF!#'
        
        ret={}        
        while True:                        
            name=F.readline()
            
            if name == '':
                break
            
            #remove newline,quotes and extension
            name=name[1:-6]
            name='_'.join(name.split('-')[2:])
            
            segs=[]
            while True:
                
                line = F.readline()
       
                if line[:-1] == '.':
                    break
                
                seg=Segment()
                
                arr=line.split(' ')
                
                seg.beg=int(arr[0])/100000
                seg.end=int(arr[1])/100000
                seg.text=arr[2]
                
                segs.append(seg)
            
            ret[name]=segs            
    return ret

class Utt:
    def __init__(self):
        self.name=''
        self.phones=[]
        self.ph_lens=[]
        self.data=None

def prepare_corp(mlf,statelist,audio_path):   
    
    states={}
    with open(statelist) as f:
        c=0
        for s in f:
            states[s[:-1]]=c
            c+=1
    
    ret=[]
    for utt_name,segs in tqdm(mlf.items()):
        
        utt=Utt()
        
        utt.name=utt_name
        
        wav_path=audio_path+'/'+utt_name+'.wav'
        
        if not os.path.exists(wav_path):
            raise IOError(wav_path)
        
        fs,utt.data=read(wav_path)
        
        assert fs == 16000
        
        for seg in segs:
            utt.phones.append(states[seg.text])
            utt.ph_lens.append(seg.end-seg.beg)
        
        ret.append(utt)
    return ret

def extract_features(corpus, savefile):
    
    mfcc=MFCC_HTK()
    h5f=h5py.File(savefile,'w')
    
    for utt in tqdm(corpus):

        feat=mfcc.get_feats(utt.data.astype(np.float64))
        delta=mfcc.get_delta(feat)
        acc=mfcc.get_delta(delta)

        feat=np.hstack((feat,delta,acc))
        utt_len=feat.shape[0]

        o=[]
        for i in range(len(utt.phones)):
            num=utt.ph_lens[i]
            o.extend([utt.phones[i]]*num)

        # here we fix an off-by-one error that happens very inrequently
        if utt_len-len(o)==1:
            o.append(o[-1])

        if len(o) != utt_len:
            print utt.name
            print len(o)
            print utt_len
            h5f.close()
        
        assert len(o)==utt_len

        g=h5f.create_group('/'+utt.name)
        
        g['in']=feat
        g['out']=o
        
        h5f.flush()
    
    h5f.close()

def normalize(corp_file):
    
    h5f=h5py.File(corp_file)

    b=0
    for utt in tqdm(h5f):
        
        f=h5f[utt]['in']
        n=f-np.mean(f)
        n/=np.std(n)        
        h5f[utt]['norm']=n
        
        h5f.flush()
        
    h5f.close()
        