"""

CODE IS AMMENDED FROM: 
https://bitbucket.org/tanya14109/cqasumm/src/master/

"""
import re
import numpy
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup

class ANSWER:
	def __init__(
		self,
		answer, 
		user_upvote_count,
		user_downvote_count,
		user_reputation,
		vote_score
		):
	
		self.answer = answer
		self.user_upvote_count = user_upvote_count
		self.user_downvote_count = user_downvote_count
		self.user_reputation = user_reputation
		self.vote_score = vote_score
		self.embedding = []

	def removeHTML(self, html):
		soup = BeautifulSoup(html, "html.parser")
		for data in soup(['style', 'script']):
			# Remove tags
			data.decompose()
		return ' '.join(soup.stripped_strings)

	def getAnswer(self):
		try:
			return self.answer
		except:
			print('Couldn\'t fetch answer')

	def getVoteScore(self):
		return self.vote_score

	def getReputation(self):
		return self.user_reputation

	def setAnswer(self,answer):
		self.answer = answer

	def getEmbedding(self):
		return self.embedding

	def setEmbedding(self,embedding):
		self.embedding = embedding

	def getWordCount(self):
		count = 0 
		for i in range(0,len(self.answer)):
			count += len(re.findall(r'\w+', self.answer[i]))
		return count


class CQA_DOCUMENT:
	def __init__(self,id,question,best_answer,all_answers):
		self.id = id
		self.question = question
		self.best_answer = best_answer
		self.num_answers = len(all_answers)
		self.reference = best_answer.getAnswer()
		self.referenceSentenceObjects  = []
		
		#Baselines
		self.KL = ''
		self.BS = ''
		self.LR = ''
		self.TR = ''
		self.OS = ''

		self.answers = all_answers

	def getQuestion(self):
		return self.question

	def getBestAnswer(self):
		return self.best_answer

	def getAllAnswers(self):
		return self.answers

	def getNumAnswers(self):
		return self.num_answers

	def getQuestionId(self):
		return self.id

	def setAllAnswers(self,all_answers):
		self.answers = all_answers

	def getReference(self):
		return self.reference

	def setReference(self,hBestAnswer):
		self.reference = hBestAnswer

	def getSentenceReferenceObjects(self):
		return self.referenceSentenceObjects

	def setSentenceReferenceObjects(self,referenceSentenceObjects):
		self.referenceSentenceObjects = referenceSentenceObjects

	def printQuestion(self):
		print('\n\n')
		print(self.question)
		print('\n\n')
		
	def setAllAnswersConcat(self):
		s = ''
		for i in range(0,self.num_answers):
			s+= '<a>'
			answer = self.answers[i].getAnswer()
			votescore = self.answers[i].getVoteScore()
			reputation = self.answers[i].getReputation()
			for j in range(len(answer)):
				s +=answer[j]
			s += "<v>%s</v>" % votescore
			s += "<r>%s</r>" % reputation
			s += "</a>"

		self.allAnswersConcat = s
		return s
		
	def setAllAnswersConcatNEW(self):
		s = []
		for i in range(0,self.num_answers):
			answer = self.answers[i].getAnswer()
			votescore = self.answers[i].getVoteScore()
			reputation = self.answers[i].getReputation()
			a = ''
			for j in range(len(answer)):
				s +=answer[j]
			dic = {
				"answer" : a,
				"vote_score" : votescore,
				"reputation" : reputation,
			}
			s.append(dic)
		self.allAnswersConcat = s
		return s


	def printAll(self):
		print('\n\n\n\n###############################################')
		print('Question',self.question)
		print('\n\n')
		print('Best Answer ',self.bestanswer.getAnswer()[0])
		print('\n\n')

		for i in range(self.num_answers):
			answer = self.answers[i].getAnswer()
			for j in range(len(answer)):
				if j>0:
					print('***')
				print(answer[j])
		
			print('........................................................')

		print('\n\n')
		print('Reference: ',self.reference)
		print('\n\n\n')

		print('KL: ',self.KL) 
		print('BS: ',self.BS)
		print('LR: ',self.LR)
		print('TR:' ,self.TR)

	def printBestAnswer(self):
		print(self.bestanswer.getAnswer())

	def getTotalWordCount(self):
		count =0
		for i in range(self.num_answers):
			count+=self.answers[i].getWordCount()

		count -= self.bestanswer.getWordCount()
		return count

	def getBestAnswer(self):
		N = self.num_answers
		correlationGraph = numpy.zeros((N,N))
		answers = self.answers

		corpus = []
		for i in range(0,N):
			corpus.append( answers.getAnswer() )
		vectorizer = TfidfVectorizer()
		X_ = vectorizer.fit_transform(corpus)
		X = X_.toarray()

		for i in range(0,N):
			for j in range(0,i):
				correlationGraph[i][j]= numpy.dot(X[i],X[j])
				correlationGraph[j][i]= numpy.dot(X[j],X[i])


		cumulativeROUGEScores = numpy.zeros((N))
		for i in range(0,N):
			for j in range(0,N):
				cumulativeROUGEScores[i]+=correlationGraph[i][j]

		ind = cumulativeROUGEScores.argmax()
		if(ind == 0):
			return 1
		else:
			return 0

	def getKL(self):
		return self.KL
	def getBS(self):
		return self.BS
	def getLR(self):
		return self.LR
	def getTR(self):
		return self.TR
	def getOS(self):
		return self.OS

	def setKL(self,text):
		self.KL = text
	def setBS(self,text):
		self.BS = text
	def setLR(self,text):
		self.LR = text
	def setTR(self,text):
		self.TR = text
	def setOS(self,text):
		self.OS = text




