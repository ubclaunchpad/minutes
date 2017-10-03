

#xml_file: file to be parsed
#sample_rate: sample rate of audio that goes with xml_file
#deliminator: deliminator for speaker labels.
#             e.g. if we have "[Michael] Something something talking blah blah", 
#                  then deliminator is a regex "\[*\]". (match open square bracket, anything, close square bracket.)
#interior: what is inside the deliminator.
#           e.g. if we have "[M] Something something talking blah blah", "[W] blah blah" and all the other speaker
#                labels are of this form, then interior is a regex "[A-Z]". (Match a single uppercase letter.)
                      
def xml_eater(xml_file, sample_rate, deliminator, interior):
    pass
