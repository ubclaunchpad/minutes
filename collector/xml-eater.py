from lxml import etree 
import numpy as np
import re
import sys

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def xml_eater(xml_file, sample_rate, deliminator, discard=None):
    """	xml_file: file to be parsed
	   	sample_rate: sample rate of audio that goes with xml_file
		deliminator: deliminator for speaker labels.
                    e.g. if we have "[M] Something something talking blah blah", 
                    then deliminator is a regex "\[[A-Z]\]". (match open square 
                    bracket, any single letter, close square bracket.)
		discard: discard any lines that contain this regex (e.g., if they label 
                    sound effects in a consistant way, we can get rid of 
		            those pretty easily using this param.)"""        
    # open xml and get xml tree	   
    tree = etree.fromstring(open(xml_file).read())
    starts = tree.xpath("text/@start")
    texts  = [x.text.strip("\n") for x in tree.xpath("text")]
    durs   = tree.xpath("text/@dur")
    num_seconds = int(float(starts[-1]) + float(durs[-1]))
    print("number of seconds * sample rate = %d" % (num_seconds*sample_rate))
    # list of all possible speakers, used to determine size and entries of np array
    speakers = []
    re_delim = re.compile(deliminator)
    re_discard = re.compile(discard)
    for text in texts: # fill speakers
        for speaker in re_delim.findall(text):
            if speaker not in speakers: speakers.append(speaker)
    print(speakers)    
    speaker_dict = {speaker: i for i, speaker in enumerate(speakers)}
    labels = np.zeros( (num_seconds*sample_rate, len(speakers)), dtype=np.int32)
    current = None	
    for n, text in enumerate(texts):
        if not bool(n % 1000): print('\r' + str(n), end="")
        sys.stdout.flush()
        start = float(starts[n])
        dur = float(durs[n])
        # [(m.start(0), m.end(0)) for m in re.finditer(pattern, string)]
        current = update_current_speaker(current, text, re_delim) # update current speaker
        if current is None: 
            pass
        elif not check_valid(text, re_delim, re_discard):
            labels[int(sample_rate*start):int(sample_rate*(start+dur)), :] = -1
        else:
            #for i in range(sample_rate*int(start*100), sample_rate*int((start+dur)*100)):
            labels[int(sample_rate*start):int(sample_rate*(start+dur)), speaker_dict[current]] = 1
    print('\r' + " "*80 + '\r', end="")
    print(labels.shape) # just making sure the array worked
    num_seconds = int(float(starts[-1]) + float(durs[-1]))
    logger.info('Result shape: {}'.format(labels.shape))
    logger.info('Expected shape: ({},{})'.format(sample_rate*num_seconds, len(speakers)))
    return labels

def check_valid(text, re_delim, re_discard):
    """there are 3 cases where text is not valid:
        (1):  multiple speakers in a text
        (2):  text doesn't start with a speaker, but has speaker in it. This
              that the start of the text has a different speaker than the end of
              the text, so there are also more than one speakers here.
        (3):  text contains sound affects deliminated by discard parameter."""  
    if len(re_delim.findall(text)) > 1: return False #(1)
    if not re_delim.match(text) and re_delim.findall(text): return False #(2)
    if re_discard.search(text): return False #(3)
    return True # default
    
def update_current_speaker(current, text, re_delim):
    """returns current speaker based on text, previous current speaker, 
        deliminator, and discard"""
    try:
        return re_delim.findall(text)[-1] # find last speaker in text
    except IndexError:
        return current

if __name__ == "__main__":
    import time
    start = time.time()
    xml_eater("sample.xml", 44000, r'\[[A-Z]\]', discard=r'\(.*\)') #tested on the file given at url in issue
    print(time.time() - start)
