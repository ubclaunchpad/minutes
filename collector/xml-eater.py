from lxml import etree 
import numpy as np
import re
#xml_file: file to be parsed
#sample_rate: sample rate of audio that goes with xml_file
#deliminator: deliminator for speaker labels.
#             e.g. if we have "[M] Something something talking blah blah", 
#                  then deliminator is a regex "\[[A-Z]\]". (match open square bracket, any single letter, close square bracket.)
#discard: discard any lines that contain this regex (e.g., if they label sound effects in a consistant way, we can get rid of 
#		those pretty easily using this param.)                      
def xml_eater(xml_file, sample_rate, deliminator, discard=None):
	# open xml
	xml_file = open(xml_file).read()
	tree = etree.fromstring(xml_file)	
	starts = tree.xpath("text/@start")
	texts  = [x.text.strip("\n") for x in tree.xpath("text")]
	durs   = tree.xpath("text/@dur") 
	speakers = [] # list of all possible speakers, used to determine size and entries of np array
	# fill speakers
	for text in texts:
		for speaker in re.findall(deliminator, text):
			if speaker not in speakers: speakers.append(speaker)
	arr = np.zeros( (1, len(speakers)), dtype=np.int64)
	current = None	
	n = 0
	# BUG: doesn't count time in between captions; TODO: fix this
	while n < len(texts):
		if check_valid(texts[n], deliminator, discard): 
			try: current = re.match(deliminator, texts[n]).group()
			except AttributeError: pass
		else:
			current = None
		newrow = [int(x==current) for x in speakers]
		start = float(starts[n])
		dur = float(durs[n])
		for i in range(int(start*100), int((start+dur)*100)):
			arr = np.vstack([arr, newrow])
		n += 1
	print(arr.shape) # just making sure the array worked

# checks if a given piece of text is spoken by one person with no sound effects
def check_valid(text, deliminator, discard=None): # -> bool:
	if discard is not None and bool(re.search(discard, text)): return False
	if len(re.findall(deliminator, text)) > 1: return False
	if not bool(re.match(deliminator, text)) and re.search(deliminator, text): return False
	return True	

#xml_eater("sample.xml", 1, r'\[[A-Z]\]', discard=r'\(.*\)') #tested on the file given at url in issue
