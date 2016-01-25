import numpy as np
import urllib
import lxml.html
import os
import tarfile
import gzip
import re
import pickle
import shutil
import random
from scikits.audiolab import Sndfile

from tqdm import *

def downloadVoxforgeData(path):
    """ Downloads the Voxforge speech database from the official website.

        Args:
            path(string): path to store the files

        Returns: nothing

        Note: this can take a long time and may require restarts. The method cehcks each 
        downloaded file, compares its size with the one online and restarts the download
        on incomplete files. You can run this method many times to make sure everything is
        downloaded correctly.
    """
    path=os.path.abspath(path)
    
    print 'Saving to '+path+'...'
    
    if not os.path.exists(path):
        os.mkdir(path)

    url='http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/'

    connection = urllib.urlopen(url)

    dom =  lxml.html.fromstring(connection.read())

    files=[]
    for link in dom.xpath('//a/@href'):
        if link.endswith('.tgz'):
            files.append(link)
            
    print 'Found '+str(len(files))+' files (sessions)...'
    
    skipped=0
    dled=0
    for f in tqdm(files):
        dl=path+'/'+f
        u=url+f

        if not os.path.exists(dl):
            dled+=1
            while True:
                try:
                    urllib.urlretrieve(u,dl)
                except urllib.ContentTooShortError as e:
                    print e.strerror
                    print 'Error downloading! Retrying...'
                    continue
                break
        else:

            dl_s=os.stat(dl).st_size
            u_p=urllib.urlopen(u)
            u_s=int(u_p.info().getheaders('Content-Length')[0])
            u_p.close()

            if u_s != dl_s:
                while True:
                    try:
                        urllib.urlretrieve(u,dl)
                    except urllib.ContentTooShortError as e:
                        print e.strerror
                        print 'Error downloading! Retrying...'
                        continue
                    break
                dled+=1
            else:
                skipped+=1
                
    print 'Downloaded '+str(dled)+' files...'
    print 'Skipped (already existing) '+str(skipped)+' files...'


class CorpusSession:
    """ A class describing a single session of the corpus.

        Properties:

            props(dictionary): properties of the session as written in the readme file
                saved as key->value dictionary

            prompts(dictionary): list of prompts (words) for individual utterances saved 
                as file(string)->prompt(string list) dictionary

            data(dictonary): list of audio recordings of the same utterances as above saved
                as file(string)->audio(numpy array) dictionary

        Note: due to files being saved in more than one format (WAV and FLAC), scikits.audiolab
        is used to load the data. Samples are stored as numpy.int16 datatype.
    """
    def __init__(self):
        self.props={}
        self.prompts={}
        self.data={}

def loadFile(path):
    """ Loads a single session from a TGZ archive.

        Args:
            path(string): path to the tgz archive
            
        Returns:
            CorpusSession: corpus session object with all the data loaded
    """
    tf=tarfile.open(path)
    
    n=tf.getnames()
    
    r_p=filter(lambda x : x.endswith('README') or x.endswith('readme'),n)[0]
    
    r=tf.extractfile(r_p)
    
    props={}
    for l in r:
        t=l.split(':')
        if(len(t)==2):
            props[t[0].strip()]=t[1].strip()
            
    props['Path']=r_p.split('/')[0]
    
    p_p=filter(lambda x : x.endswith('PROMPTS') or x.endswith('prompts'),n)[0]
    
    p=tf.extractfile(p_p)
    
    prompts={}
    for l in p:
        t=l.split()
        f=t[0].split('/')[-1]
        prompts[f]=t[1:]

    type='flac'
    if any(item.endswith('/wav') for item in n):
        type='wav'
        
    p=props['Path']
    data={}
    for f in prompts:
        try:
            ft=tf.extractfile(p+'/'+type+'/'+f+'.'+type)
        except KeyError:
            continue

        shutil.copyfileobj(ft,open('temp','wb'))
        sf=Sndfile('temp')
        data[f]=sf.read_frames(sf.nframes,dtype=np.int16)
        
    ret=CorpusSession()
    ret.props=props
    ret.prompts=prompts
    ret.data=data
    
    os.remove('temp')

    tf.close()

    return ret

def loadAudio(archive, audioname):
    """ Reads an audio file from within the archive.

        Args:
            archive(string): path to the archive (tgz)

            audioname(string): name of the file (without extension)

        Returns:
            numpy array: loaded audio signal or empty array if file not found
    """
    tf=tarfile.open(archive)
    
    n=tf.getnames()    

    ext='.flac'
    if any(item.endswith('/wav') for item in n):
        ext='.wav'
        
    p=filter(lambda x : x.endswith(audioname+ext),n)

    if len(p)==0:
        return np.array([])

    try:
        ft=tf.extractfile(p[0])
    except KeyError:
        return np.array([])

    shutil.copyfileobj(ft,open('temp','wb'))
    sf=Sndfile('temp')
    data=sf.read_frames(sf.nframes,dtype=np.int16)
        
    os.remove('temp')

    tf.close()

    return data

def loadBySpeaker(path, limit=None):
    """ Load a directory containing the Voxforge database and organize by speaker.

        Args:
            path(string): path to the folder containing the Voxforge databse (tgz files)

            limit(int): limit the number of loaded files to a given amount (in the suffled list).
                This is useful for demonstration purposes. If None (default) than it is completely
                ignored and files are read in order.

        Returns:
            dictionary: mapping speaker(string) to a dictionary of utterances(string) mapped
            to a 2-element list with the data(numpy array) and prompts(list of strings)
    """
    anon_count=0
    corp={}
    file_list=os.listdir(path)
    if limit:
        random.shuffle(file_list)
        if limit<len(file_list):
            file_list=file_list[:limit]

    for f in tqdm(file_list):
        if f.endswith('.tgz'):
            try:
                cf=loadFile(path+'/'+f)
            except IOError:
                continue
            
            if 'User Name' in cf.props:
                spk=cf.props['User Name']
            else:
                spk='anonymous'

            if spk == 'anonymous':
                anon_count+=1
                spk += '_'+str(anon_count)
            if spk in corp:
                d=corp[spk]
            else:
                d={}
            for p in cf.data:
                d[p]=[cf.data[p],cf.prompts[p]]
            corp[spk]=d
    return corp

def loadLex(path):
    """ Loads a lexicon from a file.

        Args:
            path(string): path to the lexicon

        Returns:
            dictionary: mapping a word(string) to its phonetic transcription(list of strings)

        Note: If the same word has many transcriptions, it will usually appear suffixed with 
        a number in the file. This method doesn't check if the same word exists twice in the
        lexicon and will overwrite the same word each time (leaving the last transctiption only).
    """
    path=os.path.abspath(path)

    if not os.path.exists(path):
        lex_url='http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Lexicon/VoxForge.tgz'
        urllib.urlretrieve(lex_url,path)

    tf=tarfile.open(path)

    f=tf.extractfile('VoxForge/VoxForgeDict')

    lex={}
    for l in f:
        t=re.split('\[[^\]]*\]',l)
        lex[t[0].strip()]=t[1].strip().split(' ')
        
    tf.close()

    return lex

def addPhonemesSpk(corp,lex_path):
    """ Adds phonemes to the speaker corpus. It modifies the corp object in-place to add
        a 3-rd element to the utterance list (apart from the data and prompts).

        Args:
            corp(dictionary): corpus as returned by the loadBySpeaker method

            lex_path(string): path to the lexicon file

        Returns: nothing
    """
    lex=loadLex(lex_path)
    for spk in corp.keys():
        for utt in corp[spk].keys():
            ph=[]
            for word in corp[spk][utt][1]:
                ph.extend(lex[word])
            if len(corp[spk][utt])==2:
                corp[spk][utt].append(ph)
            else:
                corp[spk][utt][2]=ph

class AliUtt:
    """ Class describing the aligned utterance.

        Properties:
            phones(list): list of phonemes(string) in the utterance

            ph_lens(list): list of the length(int) of each phoneme (in ms)

            archive(string): the name of the Voxforge archive (tgz file) containing this utterance

            audiofile(string): the name of the audio file in the archive (session)

            spk(string): the name of the speaker of the given utterance

            data(numpy array): audio data read from the file (initially None!)
    """
    def __init__(self):
        self.phones=[]
        self.ph_lens=[]
        self.archive=None
        self.audiofile=None
        self.spk=None
        self.data=None

def convertCTMToAli(ali_path,phones,audio,out):
    """ Method used to convert a CTM file generated by Kaldi into a gzipped pickle file, that
        is easier to parse in Python.

        Args:
            ali_path(string): path to the gzipped CTM file generated by Kaldi

            phones(string): path to the file with the list of phonemes (to convert string to int)

            audio(string): path to the Voxforge database sessions (tgz files)

            out(string): path of the output PKLZ file

        Returns: nothing

        Output file format:
            list: a list of AliUtt objects (without the data)
    """
    ph_map={}
    ph_count=0
    with open(phones) as f:
        for l in f:
            ph_map[l[:-1]]=ph_count
            ph_count+=1

    ali=[]
    last_utt=''
    utt=None

    print 'Reading...'
    with gzip.open(ali_path) as f:
                    
        for l in f:
            
            t=l.split(' ')
            fname=t[0]
            
            if last_utt != fname:
                
                if utt:
                    ali.append(utt)
                    
                utt=AliUtt()
                
                tt=fname.split('-')
                
                utt.spk=tt[0]
                
                if tt[0][:9] == 'anonymous':
                    tt[0]='anonymous'
                    
                utt.archive='-'.join(tt[0:3])
                utt.audiofile='-'.join(tt[3:])
                
                archive_missing=lambda x: not os.path.exists(audio+'/'+x+'.tgz')
                
                if archive_missing(utt.archive):
                    utt.archive='-'.join(tt[0:2])                    
                    utt.audiofile='-'.join(tt[2:])
                    if archive_missing(utt.archive):                        
                        raise IOError(utt.archive)
                
                last_utt=fname
            
            ph=re.sub("_[BEIS]$","",t[4]).lower()
            
            if not ph in ph_map:
                print ph
            
            assert(ph in ph_map)
            
            utt.phones.append(ph_map[ph])
            utt.ph_lens.append(int(float(t[3])*1000))
    
    if utt:
        ali.append(utt)

    print 'Writing...'
    with gzip.open(out,'wb') as f:
        pickle.dump(ali,f,pickle.HIGHEST_PROTOCOL)
        
    print 'Done'


def loadAlignedCorpus(ali_file,audio_path):
    """ Loads the data into a pickled aligned corpus.

        Args:
            ali_file(string): path to the PKLZ corpus as prepared by the convertCTMToAli method

            audio_path(string): path to the folder containing the Voxforge session (tgz files)

        Returns:
            list: updated list of aligend utterances stored as AliUtt objects (with data loaded)
    """
    with gzip.open(ali_file) as f:    
        ali=pickle.load(f)   
            
    for utt in tqdm(ali):
        data=loadAudio(audio_path+'/'+utt.archive+'.tgz',utt.audiofile)
        assert data.size > 0
        utt.data=data
            
    return ali