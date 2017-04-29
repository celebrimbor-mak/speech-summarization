'''
 * Plotting the perplexity values identified from fold run, 
 * followed by hierarchical clutering(if needed) to identify 
 * optimum tpoics, then reporting the top concepts in each topic
 *
 * @author  Arpit Mishra
 * @date 	15/04/17
 * @since   Python 2.7.6
'''
import json
import matplotlib.pyplot as plt
import numpy as np
import gensim
from gensim import corpora
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import ward, dendrogram

#Perplexity to topic plot
def combine_perplexity(source_file):
	dic = {}
	with open(source_file) as f:
		for line in f:
			temp = json.loads(line)
			for i in temp['perplexity']:
				try:	
					dic[i['parameter_value']].append(i['perplexity'])
				except:
					dic[i['parameter_value']] = [i['perplexity']]

	topics = []
	perplexity = []
	keys = dic.keys()
	keys.sort()
	for k in keys:
		topics.append(k)
		perplexity.append((sum(dic[k]))/(len(dic[k])))
		
	plt.plot(topics,perplexity)
	plt.show()

#LDA model building and returning top words for the topics
def lda(data_file, topics, words):
	doc_list = []
	with open(data_file) as f:
	    for line in f:
	        temp = json.loads(line)
	        doc_list.append(temp['tagged_terms'])
	        
	dictionary = corpora.Dictionary(doc_list)
	doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_list]
	Lda = gensim.models.ldamulticore.LdaMulticore
	ldamodel = Lda(doc_term_matrix, num_topics= topics, id2word = dictionary, workers = 4)
	topic_words = (ldamodel.print_topics(num_topics= topics, num_words= words))
	return topic_words

#Heirarchical clustering of topics
def hierarchical_clustering(topic_words):
	doc = []
	for i in topic_words:
	    comp = i[1].split('+')
	    dc = []
	    for j in comp:
	        term = j.split('*')[1]
	        term = term.replace('"','')
	        dc.append(term.strip())   
	    doc.append(dc)

	modified_doc = [' '.join(i) for i in doc]
	tf_idf = TfidfVectorizer().fit_transform(modified_doc)
	dist = 1 - cosine_similarity(tf_idf)

	linkage_matrix = ward(dist) 

	fig, ax = plt.subplots(figsize=(15, 20)) # set size
	ax = dendrogram(linkage_matrix, orientation="right");

	plt.tick_params(\
	    axis= 'x',          
	    which='both',  
	    bottom='off',     
	    top='off',         
	    labelbottom='off')

	plt.show() #show plot with tight layout
	plt.savefig('ward_clusters.png', dpi=200)
	

def driver(perplexity_file, data_file, topics, num_words, destination_file):
	# perplexity_file =  "perplexity_values.json"
	combine_perplexity(perplexity_file)
	# data_file = "tagged_data.txt"
	# topics = 100
	# words = 30
	topic_words = lda(data_file, topics, num_words)
	hierarchical_clustering(topic_words)
	final_topics = int(raw_input("Enter number of topics"))
	# final_topics = 50
	topic_words_final = lda(data_file, final_topics, num_words)

	with open(destination_file,'w') as f:
		for topic in topic_words_final:
			f.write(topic[1]+'\n')




 	