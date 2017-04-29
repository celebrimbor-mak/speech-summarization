'''
 * K-Fold training and testing LDA model to calculate perplexity
 * for identifying optimum number of topics 
 *
 * @author  Arpit Mishra
 * @date 	15/04/17
 * @since   Python 2.7.6
'''

import json
import numpy as np
import gensim
from gensim import corpora
from random import shuffle

#Creation of list of documents
def create_corpus(source_file):
	doc_list = []
	with open(source_file) as f:
		for line in f:
			temp = json.loads(line)
			doc_list.append(temp['tagged_terms'])
	return doc_list

#Division into training and test set based on number of folds, then training and testing
def fold_validation(k, perplexity_file, start, end, skip):
	doc_list = create_corpus(source_file)
	shuffle(doc_list)
	partition_value = 1 - float(1)/k
	with open(perplexity_file, 'w') as f:
		for fold in range(k):
			start_index = int(fold*(1 - partition_value)*len(doc_list))
			end_index = int((fold+1)*(1 - partition_value)*len(doc_list))
			test_corpus = doc_list[start_index:end_index]
			train_corpus = doc_list[0:start_index]
			train_corpus.extend(doc_list[end_index:])
			perplexity_values = perplexity(train_corpus, test_corpus, start, end, skip)
			f.write(json.dumps({"fold":fold, "perplexity":perplexity_values})+'\n')

#Dictionary and tdM creation
def doc_term_matrix(corpus):
	dictionary = corpora.Dictionary(corpus)
	doc_term_matrix = [dictionary.doc2bow(doc) for doc in corpus]
	return dictionary, doc_term_matrix

#Building model on training set and perplexity calculation on test set
def perplexity(train_corpus, test_corpus, start, end, skip):
	number_of_words = sum(len(term) for term in test_corpus)
	parameters = range(start, end, skip)
	train_dict, train_dtm = doc_term_matrix(train_corpus)
	test_dict, test_dtm = doc_term_matrix(test_corpus)
	perplexity_list = []
	for parameter in parameters:
		Lda = gensim.models.ldamulticore.LdaMulticore
		# ldamodel = Lda(corpus = train_corpus, workers = 2, id2word = train_dict, num_topics = parameter, iterations = 10)
		ldamodel = Lda(train_dtm, num_topics=parameter, id2word = train_dict, workers = 4)
		perplex = ldamodel.bound(test_dtm)
		perplex_per_word = np.exp2(-perplex / number_of_words)
		print parameter
		print perplex
		print perplex_per_word
		perplexity_list.append({"parameter_value":parameter, "perplexity":perplex, "perplexity_per_word": perplex_per_word})
				
	return perplexity_list

# if __name__ == '__main__':
# 	source_file = "tagged_data.txt"
# 	start, end, skip = 10, 300, 10
# 	k = 5
# 	perplexity_file = "./perplexity_values.json"
# 	fold_validation(k, perplexity_file, start, end, skip)
	


