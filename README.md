# HQBot 

## About
HQ Trivia is a game where people attempt to answer 12 questions and if all 12 questions are correct, a prize will be split between all winners. This bot helps answer questions on the app by taking a screenshot of the game on the phone and uses Google Vision to read the questions and options. It also uses Google NLP to parse the questions for meaning. It essentially automates the process of googling the questions and searching for answers. Answer choices are weighted based on how often an answer choice appears and it is left up to you to choose what answer you think is most appropriate. 

## Usage

```bash
$ git clone https://github.com/lijeffrey39/HQBot
$ cd HQ_Bot
$ pip3 install -r requirements.txt
$ python3 answer_bot.py
```

## Python Packages Used

* Google-Search-API
* wikipediaapi
* google-vision
* google-nlp 
* beautifulsoup4 

