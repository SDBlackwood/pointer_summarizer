"""

CODE IS AMMENDED FROM: 
https://bitbucket.org/tanya14109/cqasumm/src/master/

"""
import re
import nltk
import csv
import os
import numpy as np
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import brown
import operator

class sentenceImportance:
	def __init__(self, index, sentence, featureScore, nWords):
		self.index = index
		self.sentence = sentence
		self.featureScore = featureScore
		self.nWords = nWords

	def getIndex(self):
		return self.index

	def getSentence(self):
		return self.sentence

	def getFeatureScore(self):
		return self.featureScore

	def getNWords(self):
		return self.nWords

	def printAll(self):
		print('index' ,self.index) 
		print('sentence', self.sentence)
		print('featureScore', self.featureScore)
		print('nWords', self.nWords)
		print() 


def generateQuestionMap(question):
	delimiters = ['.','(',')','?','\'','!']
	stopWords = set(stopwords.words('english'))
	wordTokens = word_tokenize(question)
	filteredSentence = [w for w in wordTokens if not w in stopWords]
	filteredSentence = [w for w in filteredSentence if not w in delimiters]
	return filteredSentence

def getImportantSentenceAnswer(question,answer):
	questionWordsList = generateQuestionMap(question)
	regexPeriodEnd = re.compile('\.$|\?$|\!$')
	if not regexPeriodEnd.search(answer[0]):
		answer += '.'

	sentence = ''
	featureScore = 0
	nWords = 0
	posTaggedWords = pos_tag(word_tokenize(answer))
	sentenceImportanceList = []
	for i in range(len(posTaggedWords)):
		if(regexPeriodEnd.search(posTaggedWords[i][0])):
			sentence = sentence + posTaggedWords[i][0]
			sentenceImportanceList.append(sentenceImportance(i,sentence,featureScore,nWords))
			featureScore =0
			nWords =0
			sentence = ''
			continue
		elif(  (posTaggedWords[i][1]=='NNP') | (posTaggedWords[i][1]=='NNPS') ):
			featureScore+=1
		elif((posTaggedWords[i][1]=='NNP') | (posTaggedWords[i][1]=='NNPS')):
			featureScore+=0.25
		elif(posTaggedWords[i][0] in questionWordsList):
			featureScore+=1

		sentence = sentence + ' ' + posTaggedWords[i][0]
		nWords+=1

	sentenceImportanceList.sort(key=operator.attrgetter('featureScore'), reverse=True)

	hWordCount = 0
	index = 0
	for iter in sentenceImportanceList:
		hWordCount+=iter.getNWords()
		index+=1
		if(hWordCount>100):
			break

	sentenceImportanceList = sentenceImportanceList[:index]
	sentenceImportanceList.sort(key=operator.attrgetter('index'))

	reference = ''
	for iter in sentenceImportanceList:
		reference+= '<s>' + iter.getSentence() + '</s>'

	reference_tokens = reference.split()
	#reference_tokens = reference_tokens[:100]
	reference = ' '.join(reference_tokens)
	
	return reference,sentenceImportanceList