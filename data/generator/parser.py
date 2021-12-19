"""Parser used to get the data, create 100 word references
and return samples. 

CODE IS AMMENDED FROM: 
https://bitbucket.org/tanya14109/cqasumm/src/master/

Returns:
	[CQA_DOCUMENT]
"""
from lxml import etree
from io import StringIO, BytesIO
from document import CQA_DOCUMENT,ANSWER
from joblib import Parallel, delayed
from findImportantSentences import getImportantSentenceAnswer
from random import shuffle
import string
import re
import traceback
import pandas as pd
from bs4 import BeautifulSoup


def parseDataSet(config):
	num_questions = 0
	samples = []
	subsample = 0
	ctr = 0
	error_ctr = 0
	create_cqa = False

	remove = '<br />'
	regex = re.compile(r'('+remove+')', flags=re.IGNORECASE)
	printable = set(string.printable)


	## Get the file path from the config
	path = config.getFullPath('ai.stackexchange.com.xml')
	answer_list = []

	# FOR each post that is a question
	iterator = etree.iterparse(path, tag="data")
	for event,iter in iterator:
		for doc in iter:

			# Stop if when the limit is reached
			if ctr > config.count:
				break

			try:

				# Get non best answers
				if(doc.findtext('is_best_answer') == '0'):
    					
					if(create_cqa):
    					# Create CQA Class
						cqa = CQA_DOCUMENT(
							question_id,
							question,
							best_answer,
							answer_list
						)
						answer_list = []



						# Generating 100 word reference summary and storing it in hBestAnswer
						reference, referenceSentenceObjects = getImportantSentenceAnswer(question,best_answer.getAnswer())
						cqa.setReference(reference)
						cqa.setSentenceReferenceObjects(referenceSentenceObjects)
						samples.append(cqa)

						ctr+=1
						if ctr%100 == 0:
							print(" Hi ")
						create_cqa = False
					
					# Get meta-data about the answer.  
					user_upvote_count = doc.findtext('user_upvote_count')
					user_downvote_count = doc.findtext('user_downvote_count')
					user_reputation = doc.findtext('user_reputation')
					vote_score = doc.findtext('vote_score')

					# Filter the answer and add to the list
					answer_body = stripHTML(doc.findtext('answer_body'))
					filtered_text = [x for x in answer_body if x in printable]

					answer = ANSWER(
						filtered_text,
						user_upvote_count,
						user_downvote_count,
						user_reputation,
						vote_score
					)

					answer_list.append(answer)
					continue

				# Get best answer meta data
				user_upvote_count = doc.findtext('user_upvote_count')
				user_downvote_count = doc.findtext('user_downvote_count')
				user_reputation = doc.findtext('user_reputation')
				vote_score = doc.findtext('vote_score')
				best_answer_body = stripHTML(doc.findtext('answer_body'))
				best_answer = ANSWER(
						best_answer_body,
						user_upvote_count,
						user_downvote_count,
						user_reputation,
						vote_score
				)
				question =  doc.findtext('subject')
				question_id = doc.findtext('question_id')
				create_cqa = True

			
			except Exception as e:
				print(e)
				print(traceback.format_exc())
				error_ctr+=1

	return samples

def stripHTML(html):
	soup = BeautifulSoup(html, "html.parser")
	for data in soup(['style', 'script']):
		# Remove tags
		data.decompose()
	return ' '.join(soup.stripped_strings)