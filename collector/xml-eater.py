from lxml import etree 
import numpy as np
import re
def xml_eater(xml_file, sample_rate, deliminator, discard=None):
    """	xml_file: file to be parsed
	   	sample_rate: sample rate of audio that goes with xml_file
		deliminator: deliminator for speaker labels.
             	e.g. if we have "[M] Something something talking blah blah", 
                  then deliminator is a regex "\[[A-Z]\]". (match open square bracket, any single letter, close square bracket.)
		discard: discard any lines that contain this regex (e.g., if they label sound effects in a consistant way, we can get rid of 
		those pretty easily using this param.)"""                      
    tree = etree.fromstring(open(xml_file).read())  # open xml and get xml tree	
    starts = tree.xpath("text/@start")
    texts  = [x.text.strip("\n") for x in tree.xpath("text")]
    durs   = tree.xpath("text/@dur") 
    speakers = [] # list of all possible speakers, used to determine size and entries of np array
    # fill speakers
    for text in texts:
        for speaker in re.findall(deliminator, text):
            if speaker not in speakers: speakers.append(speaker)
    zerorow = [0 for x in speakers]
    arr = np.zeros( (1, len(speakers)), dtype=np.int64)
    current = None	
	time = 0 # time, in hundredths of seconds * sample rate
    for n, text in enumerate(texts):
        start = float(starts[n])
        dur = float(durs[n])
        while time < int(start*100)*sample_rate: # add rows of zeroes at every gap between captions
            arr = np.vstack([arr, zerorow]) 
            time += 1
        # [(m.start(0), m.end(0)) for m in re.finditer(pattern, string)]
        current = update_current_speaker(current, text, deliminator, discard) # update current speaker
        newrow = [int(x==current) for x in speakers] #1 if x matches current speaker, 0 otherwise
        for i in range(sample_rate*int(start*100), sample_rate*int((start+dur)*100)):
            arr = np.vstack([arr, newrow])
            time += 1
    print(arr.shape) # just making sure the array worked
    return arr

def update_current_speaker(current, text, deliminator, discard):
    """returns current speaker based on text, previous current speaker, deliminator, and discard"""
    current = current
    try:
        current = re.findall(text)[-1] # find last speaker in text
    except IndexError:
        pass
    return current

#xml_eater("sample.xml", 1, r'\[[A-Z]\]', discard=r'\(.*\)') #tested on the file given at url in issue
