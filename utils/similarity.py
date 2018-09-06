import json
import codecs
import gensim
import pprint
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize

pp = pprint.PrettyPrinter(indent=4)


def build_similarities(df, fecha, threshold=0.5, verbose=True):

	sim = Similarities(df)
	sims_matrix = sim.build_similarity_matrix()
	articles = sim.return_similar_articles(threshold=threshold, verbose=verbose)

	# Save results.
	print("\nSaving articles with similarities ...")
	file = codecs.open('./data/%s_04_articles.json' % fecha, 'w', encoding='utf-8')
	for article in articles:
		line = json.dumps(article, ensure_ascii=False) + "\n"
		file.write(line)
	file.close()

	return articles
	print("Done.")


class Similarities(object):

	def __init__(self, df_articles, text='text'):
		print("\nInitializing similarities ...")
		print("-"*80)
		self.df_articles = df_articles
		self.articles = df_articles[text].values
		print("Number of articles:", "{:,}".format(len(self.articles)))
		self.build_tfidf()

	def build_tfidf(self):
		"""	Train tf-idf model.	"""

		print("\nBuilding tf-idf model ...")
		print("-"*80)
		# We will now use NLTK to tokenize
		print('Word tokenization ...')
		#print(self.articles[0])
		gen_docs = [[w.lower() for w in word_tokenize(str(text))] for text in self.articles]

		# We will create a dictionary from a list of documents. A dictionary maps every word to a number.
		print("Creating dictionary ...")
		self.dictionary = gensim.corpora.Dictionary(gen_docs)
		print("Number of words in dictionary:", "{:,}".format(len(self.dictionary)))

		# Now we will create a corpus. A corpus is a list of bags of words.
		# A bag-of-words representation for a document just lists the number of times each word occurs in the document.
		print("Creating bag-of-words corpus ...")
		corpus = [self.dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]

		# Now we create a tf-idf model from the corpus. Note that num_nnz is the number of tokens.
		print("Building tf-idf model from corpus ...")
		self.tf_idf = gensim.models.TfidfModel(corpus)
		print(self.tf_idf)

		# Now we will create a similarity measure object in tf-idf space.
		print("Creating similarity measures and storing ...")
		self.sims = gensim.similarities.Similarity('./sims',
		                                      self.tf_idf[corpus],
		                                      num_features=len(self.dictionary))
		print(self.sims)
		print(type(self.sims))
		print("Done.")


	def get_similarities(self, article):
		"""	Get similarities for a given article. """

		query_doc = [w.lower() for w in word_tokenize(str(article))]
		# print(query_doc)
		query_doc_bow = self.dictionary.doc2bow(query_doc)
		# print(query_doc_bow)
		query_doc_tf_idf = self.tf_idf[query_doc_bow]
		# print(query_doc_tf_idf)

		# We show an array of document similarities to query.
		document_sims = self.sims[query_doc_tf_idf]

		return document_sims


	def build_similarity_matrix(self):
		"""
		Generates the similarty matrix by getting the individual
		similarities of each article.
		"""
		print("\nBuilding similarity matrix ...")
		print("-"*80)

		self.sims_matrix = []
		for article in self.articles:
			document_sims = self.get_similarities(article)
			self.sims_matrix.append(document_sims)

		print("Done.")

		return self.sims_matrix


	def find_similar_articles(self, threshold=0.5, verbose=True):
		"""	Looks for similar articles ...	"""

		print("\nLooking for similar items ... (threshold = %s)" % "{:.2%}".format(threshold))
		print("-"*80)
		similar_articles = []
		for idx_article, article in enumerate(self.sims_matrix):
			for idx_sim_article, similarity in enumerate(article):
				# Only interested in one combination.
				if similarity > threshold and idx_article < idx_sim_article:  # !=
					similar_articles.append([(idx_article, idx_sim_article), similarity])

		print("Similar articles:")
		pp.pprint(similar_articles)

		# Print results
		if verbose:
			print("\nChecking similar articles ...")
			for item in similar_articles:
				print("\nSimilarity:", "{:.2%}".format(item[1]))
				print("(%s)\t" % item[0][0], self.df_articles.loc[item[0][0]].title + "\n" + self.df_articles.loc[item[0][0]].url)
				print("(%s)\t" % item[0][1], self.df_articles.loc[item[0][1]].title + "\n" + self.df_articles.loc[item[0][1]].url)

		return similar_articles


	def return_similar_articles(self, threshold=0.5, verbose=True):
		"""
		Returns a list of articles in which if there are similar
		articles, the parent will be the one with higher score and the
		childs are nested under `related_articles` attribute.
		"""

		# Find similarities.
		similar_articles = self.find_similar_articles(threshold=threshold, verbose=verbose)

		# Creating array of parent articles (with childs nested).
		articles = []
		print(self.df_articles)
		print(self.df_articles.iloc[1])

		"""

		CONTINUE HERE --> try to understand why it is not working...

		"""




		for sim_tuple in similar_articles:
			# try:
			print(sim_tuple[0])
			print(sim_tuple[0][0])
			print(self.df_articles.iloc[sim_tuple[0][0]])
			principal_article = self.df_articles.iloc[sim_tuple[0][0]].to_dict()
			principal_article['id'] = sim_tuple[0][0]
			child_article = self.df_articles.iloc[sim_tuple[0][1]].to_dict()
			child_article['id'] = sim_tuple[0][1]
			articles_saved = [i['url'] for i in articles]
			if principal_article['url'] in articles_saved:
				articles[-1]['related_articles'].append(child_article)
			else:
				principal_article['related_articles'] = []
				principal_article['related_articles'].append(child_article)
				articles.append(principal_article)
			# delete child articles.
			self.df_articles.drop(self.df_articles.iloc[[sim_tuple[0][1]]].index, inplace=True)
			# except:
			# 	pass

		print(articles)

		# Delete all parent articles.
		for sim_tuple in similar_articles:
			try:
				self.df_articles.drop(self.df_articles.iloc[[sim_tuple[0][0]]].index, inplace=True)
			except:
				pass

		# Print dependency tree.
		print('--> Printing dependency tree...')
		for article in articles:
			print("\nParent: (%s) %s" % (article['id'], article['title']))
			for related_article in article['related_articles']:
				print(" |__ Child: (%s) %s" % (related_article['id'], related_article['title']))

		# Add remaining articles.
		for idx, row in self.df_articles.iterrows():
			article = row.to_dict()
			article['related_articles'] = []
			#print("Saving article: %s" % article['title'])
			articles.append(article)

		# Sort articles by score.
		articles = sorted(articles, key=lambda x: x['score'], reverse=True)

		return articles
