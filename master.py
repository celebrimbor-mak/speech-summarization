'''
 * This function controls all individual
 * modules of the code
 *
 * @author  Arpit Mishra
 * @date 	15/04/17
 * @since   Python 2.7.6
'''

import os
from audio_to_text import convert
from tag import tag_terms
from lda import driver
from topic_model_k_fold import fold_validation

if __name__ == '__main__':

	#Conversion of audio to text
	location = "./data/"
	convert(location)
	os.chdir('../')

	#POS tagging and term extraction
	location = './data/'
	destination_file = 'tagged_document.txt'
	tag_terms(location, destination_file)

	#Number of topics identification
	source_file = "tagged_data.txt"
	start, end, skip = 10, 300, 10
	k = 5
	perplexity_file = "./perplexity_values.json"
	fold_validation(k, perplexity_file, start, end, skip)

	#Hierarchical clustering and Topic Modeling
	perplexity_file =  "perplexity_values.json"
	data_file = "tagged_data.txt"
	topics = 100
	num_words = 30
	destination_file = "top_topics.txt"
	driver(perplexity_file, data_file, topics, num_words, destination_file)

	
