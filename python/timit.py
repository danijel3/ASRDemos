import sys
import numpy as np
from scipy.io.wavfile import read
import os
import h5py
from tqdm import *

sys.path.append('../PyHTK/python')

from HTKFeat import MFCC_HTK
from PHN import PHN

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

def prepare_corp_dir(list_file,path,win_len=0.025,win_shift=0.01):

    with open(list_file) as f:
        file_list=f.read().splitlines()

    ret=[]
    for f in tqdm(file_list):
        utt=Utt()
       
        utt.name=f
        
        fs,utt.data=read(path+'/'+f+'.wav')
        
        assert fs == 16000

        tg_file=path+'/'+f+'.phn'

        if not os.path.exists(tg_file):
            raise IOError(tg_file)
        
        phn=PHN()
        phn.load(tg_file)

        win_len_s=win_len*fs
        win_shift_s=win_shift*fs

        win_num=np.floor((utt.data.size-win_len_s)/win_shift_s).astype('int')+1

        if phn.segments[0].xmin > 0:
            phn.segments[0].xmin = 0

        if phn.segments[-1].xmax < utt.data.size:
            phn.segments[-1].xmax = utt.data.size

        seq=phn.toSequence(win_num,win_shift_s,win_len_s)

        fixes={'h\\#':'h#','ax-h':'axh'}

        lc=-1
        for ph in seq:
            if ph in fixes:
                ph = fixes[ph]
            if ph not in timit61:
                raise RuntimeError('Error in file '+f)
            c=timit61.index(ph)
            if c!=lc:
                utt.phones.append(c)
                utt.ph_lens.append(1)
                lc=c
            else:
                utt.ph_lens[-1]+=1

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



#this method demonstrates how the arrays below were generated
#it doesn't need to be run manually
#the statelist can be found in the CNTK project at https://github.com/Microsoft/CNTK
#under "CNTK/Examples/Speech/Miscellaneous/TIMIT/lib/mlf/"
#warning: the r61t39 variable was defined manually (see below)
def loadReductions(statelist):
    with open(statelist) as f:
        timit183=f.read().splitlines()

    r183t61={}
    timit61=set()
    for s in timit183:
        r=s.split('_')[0]
        r183t61[s]=r
        timit61.add(r)
    timit61=list(timit61)

    r183t61id=[]
    for s in timit183:
        r=r183t61[s]
        r183t61id.append(timit61.index(r))

    timit39=set()
    for p in timit61:
        if p in r61t39:
            r=r61t39[p]
            if len(r)>0:
                timit39.add(r)
        else:
            timit39.add(p)
    timit39=list(timit39)

r183t61={'m_s4': 'm', 'ch_s2': 'ch', 'ch_s3': 'ch', 'ch_s4': 'ch', 'ux_s4': 'ux', 
    'm_s3': 'm', 'bcl_s4': 'bcl', 's_s4': 's', 's_s3': 's', 's_s2': 's', 'oy_s3': 'oy', 
    'oy_s2': 'oy', 'oy_s4': 'oy', 'uh_s2': 'uh', 'uh_s3': 'uh', 'bcl_s3': 'bcl',
    'uh_s4': 'uh', 'pau_s2': 'pau', 'pau_s3': 'pau', 'ah_s4': 'ah', 'bcl_s2': 'bcl', 
    'ah_s2': 'ah', 'ah_s3': 'ah', 'pau_s4': 'pau', 'sh_s3': 'sh', 'b_s4': 'b', 
    'dx_s2': 'dx', 'ax_s4': 'ax', 'ax_s2': 'ax', 'ax_s3': 'ax', 'b_s2': 'b', 'b_s3': 'b',
    'k_s4': 'k', 'k_s3': 'k', 'k_s2': 'k', 'gcl_s2': 'gcl', 'gcl_s3': 'gcl', 
    'gcl_s4': 'gcl', 'z_s3': 'z', 'ix_s4': 'ix', 'ix_s2': 'ix', 'ix_s3': 'ix',
    'sh_s2': 'sh', 'axh_s3': 'axh', 'axh_s2': 'axh', 'y_s3': 'y', 'y_s2': 'y', 
    'p_s4': 'p', 'axh_s4': 'axh', 'en_s2': 'en', 'en_s3': 'en', 'en_s4': 'en', 
    'p_s2': 'p', 'l_s2': 'l', 'l_s3': 'l', 'l_s4': 'l', 't_s2': 't', 'aa_s4': 'aa', 
    'aa_s3': 'aa', 'aa_s2': 'aa', 'w_s3': 'w', 'w_s2': 'w', 'w_s4': 'w', 'q_s3': 'q', 
    'q_s2': 'q', 'sh_s4': 'sh', 'q_s4': 'q', 'h#_s4': 'h#', 'h#_s2': 'h#', 
    'h#_s3': 'h#', 't_s4': 't', 'r_s4': 'r', 'zh_s3': 'zh', 'zh_s2': 'zh', 
    'zh_s4': 'zh', 'r_s2': 'r', 'r_s3': 'r', 'g_s3': 'g', 'g_s2': 'g',
    'ow_s3': 'ow', 'g_s4': 'g', 'ow_s4': 'ow', 'z_s4': 'z', 't_s3': 't',
    'y_s4': 'y', 'ao_s3': 'ao', 'ao_s2': 'ao', 'ao_s4': 'ao', 'aw_s3': 'aw',
    'aw_s2': 'aw', 'pcl_s4': 'pcl', 'aw_s4': 'aw', 'nx_s4': 'nx', 'axr_s4': 'axr', 
    'axr_s3': 'axr', 'axr_s2': 'axr', 'ow_s2': 'ow', 'epi_s2': 'epi', 
    'epi_s3': 'epi', 'epi_s4': 'epi', 'uw_s4': 'uw', 'uw_s3': 'uw', 'uw_s2': 'uw', 
    'ay_s4': 'ay', 'eh_s2': 'eh', 'eh_s3': 'eh', 'hv_s3': 'hv', 'hv_s2': 'hv', 
    'hv_s4': 'hv', 'ay_s3': 'ay', 'ay_s2': 'ay', 'v_s2': 'v', 'v_s3': 'v',
    'v_s4': 'v', 'dcl_s3': 'dcl', 'dcl_s2': 'dcl', 'eh_s4': 'eh', 
    'ng_s2': 'ng', 'dcl_s4': 'dcl', 'eng_s2': 'eng', 'eng_s3': 'eng', 
    'eng_s4': 'eng', 'dx_s3': 'dx', 'nx_s3': 'nx', 'jh_s3': 'jh',
    'jh_s2': 'jh', 'jh_s4': 'jh', 'el_s4': 'el', 'el_s2': 'el', 
    'el_s3': 'el', 'f_s2': 'f', 'f_s3': 'f', 'f_s4': 'f', 'p_s3': 'p', 
    'tcl_s4': 'tcl', 'm_s2': 'm', 'dx_s4': 'dx', 'ng_s4': 'ng',
    'd_s2': 'd', 'd_s3': 'd', 'hh_s4': 'hh', 'hh_s3': 'hh', 'hh_s2': 'hh',
    'd_s4': 'd', 'nx_s2': 'nx', 'tcl_s3': 'tcl', 'tcl_s2': 'tcl', 'n_s2': 'n',
    'n_s3': 'n', 'n_s4': 'n', 'iy_s4': 'iy', 'em_s4': 'em', 'ey_s3': 'ey', 
    'ey_s2': 'ey', 'z_s2': 'z', 'ey_s4': 'ey', 'em_s3': 'em', 'em_s2': 'em', 
    'kcl_s4': 'kcl', 'kcl_s2': 'kcl', 'kcl_s3': 'kcl', 'dh_s3': 'dh', 'dh_s2': 'dh', 
    'ng_s3': 'ng', 'pcl_s3': 'pcl', 'pcl_s2': 'pcl', 'dh_s4': 'dh', 'iy_s3': 'iy', 
    'iy_s2': 'iy', 'ae_s3': 'ae', 'ae_s2': 'ae', 'ae_s4': 'ae', 'th_s3': 'th',
    'th_s2': 'th', 'th_s4': 'th', 'ux_s2': 'ux', 'ux_s3': 'ux', 'er_s4': 'er', 
    'ih_s4': 'ih', 'ih_s2': 'ih', 'ih_s3': 'ih', 'er_s2': 'er', 'er_s3': 'er'} 

# warning - q is reduced to blank!
# this was manually copied from:
# Santiago Fernandez, Alex Graves, Jurgen Schmidhuber: "Phoneme recognition in TIMIT with BLSTM-CTC"
# http://arxiv.org/pdf/0804.3269.pdf
r61t39={"pcl":"sil","tcl":"sil","kcl":"sil","nx":"n","pau":"sil","ao":"aa","ax":"ah",
    "ix":"ih","bcl":"sil","gcl":"sil","dcl":"sil","q":"","em":"m","en":"n","eng":"ng",
    "zh":"sh","el":"l","h#":"sil","h\\#":"sil","epi":"sil","hv":"hh","ux":"uw","axr":"er",
    "ax-r":"er","ax-h":"ah","axh":"ah"}


r183t61id=np.array([0, 0, 0, 3, 3, 3, 6, 6, 6, 7, 7, 7, 13, 13, 13, 16, 16, 16, 58, 58, 58, 
    41, 41, 41, 15, 15, 15, 42, 42, 42, 28, 28, 28, 2, 2, 2, 43, 43, 43, 29, 29, 29, 31, 31, 
    31, 38, 38, 38, 4, 4, 4, 1, 1, 1, 39, 39, 39, 11, 11, 11, 20, 20, 20, 56, 56, 56, 18, 
    18, 18, 12, 12, 12, 45, 45, 45, 44, 44, 44, 21, 21, 21, 14, 14, 14, 36, 36, 36, 34, 34, 
    34, 9, 9, 9, 5, 5, 5, 24, 24, 24, 37, 37, 37, 46, 46, 46, 32, 32, 32, 48, 48, 48, 47, 47, 
    47, 49, 49, 49, 22, 22, 22, 23, 23, 23, 57, 57, 57, 55, 55, 55, 51, 51, 51, 19, 19, 19, 
    26, 26, 26, 50, 50, 50, 53, 53, 53, 52, 52, 52, 25, 25, 25, 54, 54, 54, 10, 10, 10, 30, 
    30, 30, 27, 27, 27, 60, 60, 60, 40, 40, 40, 33, 33, 33, 8, 8, 8, 35, 35, 35, 59, 59, 59, 
    17, 17, 17])

timit183=['aa_s2', 'aa_s3', 'aa_s4', 'ae_s2', 'ae_s3', 'ae_s4', 'ah_s2', 'ah_s3', 
    'ah_s4', 'ao_s2', 'ao_s3', 'ao_s4', 'aw_s2', 'aw_s3', 'aw_s4', 'ax_s2', 'ax_s3', 
    'ax_s4', 'axh_s2', 'axh_s3', 'axh_s4', 'axr_s2', 'axr_s3', 'axr_s4', 'ay_s2', 
    'ay_s3', 'ay_s4', 'b_s2', 'b_s3', 'b_s4', 'bcl_s2', 'bcl_s3', 'bcl_s4', 'ch_s2', 
    'ch_s3', 'ch_s4', 'd_s2', 'd_s3', 'd_s4', 'dcl_s2', 'dcl_s3', 'dcl_s4', 'dh_s2', 
    'dh_s3', 'dh_s4', 'dx_s2', 'dx_s3', 'dx_s4', 'eh_s2', 'eh_s3', 'eh_s4', 'el_s2', 
    'el_s3', 'el_s4', 'em_s2', 'em_s3', 'em_s4', 'en_s2', 'en_s3', 'en_s4', 'eng_s2', 
    'eng_s3', 'eng_s4', 'epi_s2', 'epi_s3', 'epi_s4', 'er_s2', 'er_s3', 'er_s4', 'ey_s2', 
    'ey_s3', 'ey_s4', 'f_s2', 'f_s3', 'f_s4', 'g_s2', 'g_s3', 'g_s4', 'gcl_s2', 'gcl_s3', 
    'gcl_s4', 'h#_s2', 'h#_s3', 'h#_s4', 'hh_s2', 'hh_s3', 'hh_s4', 'hv_s2', 'hv_s3', 
    'hv_s4', 'ih_s2', 'ih_s3', 'ih_s4', 'ix_s2', 'ix_s3', 'ix_s4', 'iy_s2', 'iy_s3', 
    'iy_s4', 'jh_s2', 'jh_s3', 'jh_s4', 'k_s2', 'k_s3', 'k_s4', 'kcl_s2', 'kcl_s3', 
    'kcl_s4', 'l_s2', 'l_s3', 'l_s4', 'm_s2', 'm_s3', 'm_s4', 'n_s2', 'n_s3', 'n_s4', 
    'ng_s2', 'ng_s3', 'ng_s4', 'nx_s2', 'nx_s3', 'nx_s4', 'ow_s2', 'ow_s3', 'ow_s4', 
    'oy_s2', 'oy_s3', 'oy_s4', 'p_s2', 'p_s3', 'p_s4', 'pau_s2', 'pau_s3', 'pau_s4', 
    'pcl_s2', 'pcl_s3', 'pcl_s4', 'q_s2', 'q_s3', 'q_s4', 'r_s2', 'r_s3', 'r_s4', 
    's_s2', 's_s3', 's_s4', 'sh_s2', 'sh_s3', 'sh_s4', 't_s2', 't_s3', 't_s4', 'tcl_s2', 
    'tcl_s3', 'tcl_s4', 'th_s2', 'th_s3', 'th_s4', 'uh_s2', 'uh_s3', 'uh_s4', 'uw_s2', 
    'uw_s3', 'uw_s4', 'ux_s2', 'ux_s3', 'ux_s4', 'v_s2', 'v_s3', 'v_s4', 'w_s2', 'w_s3', 
    'w_s4', 'y_s2', 'y_s3'  , 'y_s4', 'z_s2', 'z_s3', 'z_s4', 'zh_s2', 'zh_s3', 'zh_s4']

timit61=['aa', 'el', 'ch', 'ae', 'eh', 'ix', 'ah', 'ao', 'w', 'ih', 'tcl', 'en', 'ey', 
    'aw', 'h#', 'ay', 'ax', 'zh', 'er', 'pau', 'eng', 'gcl', 'ng', 'nx', 'iy', 'sh', 'pcl', 
    'uh', 'bcl', 'dcl', 'th', 'dh', 'kcl', 'v', 'hv', 'y', 'hh', 'jh', 'dx', 'em', 'ux',
    'axr', 'b', 'd', 'g', 'f', 'k', 'm', 'l', 'n', 'q', 'p', 's', 'r', 't', 'oy', 'epi', 
    'ow', 'axh', 'z', 'uw']

timit39=['aa', 'iy', 'ch', 'ae', 'eh', 'ah', 'ih', 'ey', 'aw', 'ay', 'er', 'ng', 'r', 'th', 
    't', 'sil', 'oy', 'dh', 'ow', 'hh', 'jh', 'dx', 'b', 'd', 'g', 'f', 'uw', 'm', 'l', 'n', 'p',
    's', 'sh', 'uh', 'w', 'v', 'y', 'z', 'k']


def reduce183to61id(ids):
    return r183t61id[ids]

def reduce183to61idseq(idseq):
    ret=[]
    for s in idseq:
        ret.append(r183t61id[s])
    return np.array(ret)

def reduce61to39(seq):
    ret=[]
    for s in seq:
        if s in r61t39:
            r=r61t39[s]
            if len(r)>0:
                ret.add(r)
    else:
        ret.add(s)
    return ret