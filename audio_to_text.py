'''
 * Using Google API to convert audio to text
 * @author  Arpit Mishra
 * @date 	14/04/17
 * @since   Python 2.7.6
'''
import speech_recognition as sr
import sys
import glob
import os
def convert(location):
	os.chdir(location)
	for filename in glob.glob('*.wav'):
		print filename
		r = sr.Recognizer()
		with sr.AudioFile(filename) as source:
			audio = r.record(source)

		try:
			text = r.recognize_google(audio)
			with open(filename[:-4]+'.trn', 'w') as f:
				f.write(text)
		except Exception as e:
			print e

# if __name__ == '__main__':
# 	location = "./"
# 	convert(location)
