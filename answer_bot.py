'''

	TODO:
	* Implement normalize func
	* Attempt to google wiki \"...\" part of question
	* Rid of common appearances in 3 options
	* Automate screenshot process

'''

import json
import urllib2
from bs4 import BeautifulSoup
from bs4.element import Comment
from google import google
from PIL import Image
import pytesseract
import argparse
import cv2
import os
import pyscreenshot as Imagegrab
import sys
from halo import Halo
import io
import multiprocessing
import time
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "mTurk-Firebase-59683a14dad8.json"

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types

from socketIO_client_nexus import SocketIO, LoggingNamespace

def on_bbb_response(*args):
    print('lol', args)

# Instantiates a client
clientL = language.LanguageServiceClient()

# Instantiates a client
client = vision.ImageAnnotatorClient()

# sample questions from previous games
sample_questions = {}

# list of words to clean from the question during google search
remove_words = []

# negative words
negative_words= []

# load sample questions
def load_json():
	global remove_words, sample_questions, negative_words
	remove_words = json.loads(open("Data/settings.json").read())["remove_words"]
	negative_words = json.loads(open("Data/settings.json").read())["negative_words"]
	sample_questions = json.loads(open("Data/question.json").read())


# take screenshot of question
def screen_grab(to_save):
	# 31,228 485,620 co-ords of screenshot// left side of screen
	im = Imagegrab.grab(bbox=(31,228,485,580))
	im.save(to_save)

# get OCR text //questions and options
def read_screen():
	spinner = Halo(text='Reading screen ', spinner='bouncingBar')
	spinner.start()
	screenshot_file="Screens/to_ocr.png"
	screen_grab(screenshot_file)

	#prepare argparse
	ap = argparse.ArgumentParser(description='HQ_Bot')
	ap.add_argument("-i", "--image", required=False,default=screenshot_file,help="path to input image to be OCR'd")
	ap.add_argument("-p", "--preprocess", type=str, default="thresh", help="type of preprocessing to be done")
	args = vars(ap.parse_args())

	# load the image
	image = cv2.imread(args["image"])
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	if args["preprocess"] == "thresh":
		gray = cv2.threshold(gray, 0, 255,
			cv2.THRESH_TOZERO | cv2.THRESH_OTSU)[1]
		gray = cv2.medianBlur(gray, 3)
	elif args["preprocess"] == "blur":
		gray = cv2.medianBlur(gray, 3)

	# store grayscale image as a temp file to apply OCR
	filename = "Screens/{}.png".format(os.getpid())
	cv2.imwrite(filename, gray)

	with io.open(filename, 'rb') as image_file:
		content = image_file.read()

	image = vision.types.Image(content=content)

	response = client.document_text_detection(image=image)
	document = response.full_text_annotation

	blocks = []
	for page in document.pages:
		for block in page.blocks:
			block_words = []
			for paragraph in block.paragraphs:
				block_words.extend(paragraph.words)

			block_symbols = []
			block_text = ''
			for word in block_words:
				block_symbols.extend(word.symbols)
				word_text = ''
				for symbol in word.symbols:
					word_text = word_text + symbol.text

				block_text += ' ' + word_text

			blocks.append(block_text)

	# load the image as a PIL/Pillow image, apply OCR, and then delete the temporary file
	text = pytesseract.image_to_string(Image.open(filename))
	os.remove(filename)
	# os.remove(screenshot_file)

	spinner.succeed()
	spinner.stop()
	return text, blocks

# get questions and options from OCR text
def parse_question():
	text, blocks = read_screen()
	if (len(blocks) == 4):
		return blocks[0], blocks[1:]

	for i in range(len(blocks)):
		blocks[i] = blocks[i].encode('utf-8')

	lines = text.splitlines()
	question = ""
	options = list()
	flag = False

	for line in lines :
		if not flag :
			question = question+" "+line

		if '?' in line :
			flag = True
			continue

		if flag :
			if line != '' :
				options.append(line)

	return question, options


# simplify question and remove which,what....etc //question is string
def simplify_ques(question):
	neg = False
	qwords = question.lower().split()
	if [i for i in qwords if i in negative_words]:
		neg=True

	qwords = question.lower().split()

	cleanwords = [word for word in qwords if word.lower() not in remove_words]
	temp = ' '.join(cleanwords)
	clean_question=""

	#remove ?
	for ch in temp:
		if ch!="?" or ch!="\"" or ch!="\'":
			clean_question=clean_question+ch

	return clean_question.lower(), neg


# get web page
def get_page(link):
	try:
		if link.find('mailto') != -1:
			return ''
		req = urllib2.Request(link, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64)'})
		html = urllib2.urlopen(req).read()
		return html
	except (urllib2.URLError, urllib2.HTTPError, ValueError) as e:
		return ''


# normalize points // get rid of common appearances // "quote" wiki option + ques
def normalize():
	return None

# take screen shot of screen every 2 seconds and check for question
def check_screen():
	return None

# wait for certain milli seconds
def wait(msec):
	return None

# answer by combining two words
def smart_answer(content, qwords):
	points = 0

	for i in range(len(qwords) - 1):
		num = content.count(qwords[i] + " " + qwords[i + 1])
		points += (200 * num)

	return points


def getPage(link):
	content = get_page(link)
	soup = BeautifulSoup(content,"lxml")
	texts = soup.findAll(text=True)
	visible_texts = filter(tag_visible, texts)
	page = u" ".join(t.strip() for t in visible_texts).lower()
	return page


def getPoints(link, words):
	page = getPage(link)
	temp = 0

	for word in words:
		temp = temp + page.count(word[0])
		#temp = temp + (page.count(word[0]) * word[1]) #account for salience

	#temp += smart_answer(page, words)
	return temp


def searchQuestionPoint(page, option):
	oWords = option.lower().split()
	temp = 0
	temp = page.count(option)

	return (temp * 1000)


def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def optionPoint((o, words, num_pages, question)):
	if (question == False):
		options = words[0]
		searchWiki = google.search(words[1], num_pages) #search question

		# print("getPage")
		page = getPage(searchWiki[0].link)

		pointsOptions = [0, 0, 0]
		for i in range(len(options)):
			pointsOptions[i] += searchQuestionPoint(page, options[i])
		# print(pointsOptions)

		return pointsOptions
	else:
		o = o.lower()
		o += ' wiki'

		# get google search results for option + 'wiki'
		search_wiki = google.search(o, num_pages)

		temp = 0
		for i in range(len(search_wiki) / 4):
			temp += getPoints(search_wiki[i].link, words)

		return temp


# use google to get wiki page
def google_wiki(question, words, options):
	spinner = Halo(text='Googling and searching Wikipedia', spinner='dots2')
	spinner.start()
	num_pages = 1
	content = ""

	pool = multiprocessing.Pool(processes = 4, maxtasksperchild = 1)

	newOptions = map(lambda o: (o, words, num_pages, True), options)
	newOptions.append(("", [options, question, ""], num_pages, False))
	result = pool.map(optionPoint, newOptions)

	numsAdd = result[3]

	result = [result[0] + numsAdd[0], result[1] + numsAdd[1], result[2] + numsAdd[2]]

	# print(result)
	# pool.close()
	# pool.join()

	# print("terminate")
	pool.terminate()

	# print("pool")

	spinner.succeed()
	spinner.stop()

	return result


def printResult(question, options):
	points = []

	simq = ""
	points = []
	simq, neg = simplify_ques(question)

	options = map(lambda s: s.encode("utf-8").strip().lower(), options)


	# Instantiates a plain text document.
	document = types.Document(
	content = question,
	type = enums.Document.Type.PLAIN_TEXT)

	# Detects entities in the document.
	entities = clientL.analyze_entities(document).entities

	# entity types from enums.Entity.Type
	entity_type = ('UNKNOWN', 'PERSON', 'LOCATION', 'ORGANIZATION',
	'EVENT', 'WORK_OF_ART', 'CONSUMER_GOOD', 'OTHER')

	words = []
	for entity in entities:
		words.append([entity.name, entity.salience])

	# print(question)
	# print(options)
	# print("\n")

	points = google_wiki(question, words, options)
	print("\n" + question + "\n")

	pointToChoose = 0
	if (neg):
		pointToChoose = min(points)
	else:
		pointToChoose = max(points)

	optionI = 0
	x = 0
	for point, option in zip(points, options):

		if (point == pointToChoose):
			optionI = x
			print("vvvvvvvvvvvvvvvvvvvvvv")
		print(option + " { points: " + str(point) + " }")
		print("\n")
		x = x + 1

	with SocketIO('localhost', 3000, LoggingNamespace) as socketIO:
		socketIO.emit('answer', {"option": optionI, "points": points}, on_bbb_response)

		socketIO.wait_for_callbacks(seconds=1)

# return points for sample_questions
def get_points_sample():
	start = time.time()

	simq = ""
	x = 0
	for question in sample_questions:
		x = x + 1
		options = sample_questions[question]

		with SocketIO('localhost', 3000, LoggingNamespace) as socketIO:
		    socketIO.emit('new question', {'q': question,
				'c1': options[0], 'c2': options[1], 'c3': options[2]}, on_bbb_response)

		    socketIO.wait_for_callbacks(seconds=1)


		start = time.time()
		printResult(question, options)
		end = time.time()
		print(end - start)


# return points for live game // by screenshot
def get_points_live():
	question, options = parse_question()


	with SocketIO('localhost', 3000, LoggingNamespace) as socketIO:
		socketIO.emit('new question', {'q': question,
			'c1': options[0], 'c2': options[1], 'c3': options[2]}, on_bbb_response)

		socketIO.wait_for_callbacks(seconds=1)

	start = time.time()
	printResult(question, options)
	end = time.time()
	print(end - start)


# menu// main func
if __name__ == "__main__":
	load_json()
	while(1):
		keypressed = raw_input('\nPress s to screenshot live game, sq to run against sample questions or q to quit:\n')
		if keypressed == 's':
			get_points_live()
		elif keypressed == 'sq':
			get_points_sample()
		elif keypressed == 'q':
			break
		else:
			print("\nUnknown input")
