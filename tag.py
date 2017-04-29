'''
 * POS tagging followed by capturing Nouns and Verbs
 *
 * @author  Arpit Mishra
 * @date 	15/04/17
 * @since   Python 2.7.6
'''
import nltk
import glob
import json

def tag_terms(location, destination_file):
	is_useful = lambda pos: pos[:2] == 'NN' or pos[:2] == 'VB'
	with open(destination_file, 'w') as writer:
		for filename in glob.glob(location+'*.trn'):
			with open(filename) as f:
				tagged_terms = []
				for line in f:
					tokenized = nltk.word_tokenize(line.lower())
					required_words =  [word for (word, pos) in nltk.pos_tag(tokenized) if is_useful(pos)]
					tagged_terms.extend(required_words)
				writer.write(json.dumps({'doc':filename[8:-8], 'tagged_terms':tagged_terms})+'\n')


# if __name__ == '__main__':
# 	location = '../data/'
# 	destination_file = 'tagged_document.txt'
# 	tag_terms(location, destination_file)

