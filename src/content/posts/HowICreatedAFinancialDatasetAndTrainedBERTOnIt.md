---
title: How I Created A Financial Dataset And Trained BERT On It
author: Ugur Sadiklar
date: 2024-03-07
tags: [bert, huggingface, finance, machine learning, nlp, python, jupyter]
---

### Introduction

For my bachlor thesis I wanted to create a dataset of sentences from Yahoo Finance news articles and train a pre-trained BERT model on it. In this post, I will explain all the steps it took to create the dataset and train the model.

### How I created the dataset

First, I decided to only use articles in the Business category.

Second, I analyzed the HTML structure of a Yahoo Finance news article page. I found that the article text is inside a div with the class "caas-body".

```html
<div class="caas-body">...</div>
```

Inside this div there are multiple p tags that contain the sentences of the article like:

```html
<p>
	The Icheon-based firm is investing more than $1 billion in South Korea to expand and improve the
	final steps of its chip manufacture, said Lee Kang-Wook, a former Samsung Electronics Co. engineer
	who now heads up packaging development at SK Hynix. Innovation with that process is at the heart
	of HBM’s advantage as the most sought-after AI memory, and further advances will be key to
	reducing power consumption, driving performance and cementing the company’s lead in the HBM
	market.
</p>
```

After that I started to create a notebook to scrape the text from the articles and save them to an SQLite database. For the scraping part I used the BeautifulSoup library. In the following code snippets I will shortly explain how I did it:

```python
res = requests.get(link) # link is the URL of the article
soup = BeautifulSoup(res.text, "html.parser") # res.text is the HTML of the article
# store the title in a variable to save it in the database
title = soup.find("title").text

# The title Yahoo is returned on the html, when a 404 error occurs
if(soup == None or title == 'Yahoo'):
    continue

# get the div with the class caas-body
soup = soup.find("div", attrs={'class':'caas-body'})

for div in soup.find_all('div'):
    div.unwrap() # remove all div tags

for ul in soup.find_all('ul'):
    p_tag = ul.find_previous_sibling('p') # get the p tag before the ul tag

    if p_tag:
        p_tag.decompose() # remove the p tag

# remove all ul tags (these are usually lists of links to other articles)
for ul in soup.find_all("ul"):
    ul.decompose()

for button in soup.find_all("button"):
    button.decompose() # remove all button tags

for p_tag in soup.find_all('p', string="©2023 Bloomberg L.P."):
    p_tag.decompose() # remove the p tag with the copyright text

text = ""
for p in soup.find_all('p'):
    text = text + p.text + "\n"
```

This scraping script is very rudimentary and only works for the specific HTML structure of Yahoo Finance news articles. For example it does not work for articles that contain images or videos. Also it is not able to extract only the useful sentences from the articles, because they are so dynamic in their structure. But for my purpose it was enough.

Now I needed to find a way to automate the scraping of link articles to feed into this script. I went for the Selenium library to automate this process in a headless browser. I created another notebook that opens a Chrome browser, scrolls down on the cookie modal, clicks deny, then scrolls down to the bottom of the page to load more articles. After that it extracts the HTML of the page. When this is done, BeautifulSoup is used to extract the links of the articles and save them to a list. This list is then used to feed the scraping script.

This code opens a Chrome browser and gets the HTML of the Yahoo Finance news page:

```python
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time

# Initialisiere den WebDriver
driver = webdriver.Chrome()

# Navigiere zur Webseite
driver.get("https://finance.yahoo.com/news")

# Ablehnen des Cookie Popups
scroll_button = driver.find_element(By.ID, "scroll-down-btn")
scroll_button.click()
cookie_button = driver.find_element(By.CLASS_NAME, "reject-all")
cookie_button.click()

html_element = driver.find_element(By.TAG_NAME, 'html')

# Drücke 16 Mal die "End"-Taste, um ans Ende der Seite zu scrollen (dynamisch geladenes HTML)
for _ in range(16):
    html_element.send_keys(Keys.END)
    time.sleep(1)

# Hole das HTML der Seite
html = driver.page_source

# Schließe den Browser
driver.quit()
```

After that the following script saves the links from the HTML to a file:

```python
# Verwende BeautifulSoup, um das HTML zu parsen
soup = BeautifulSoup(html, 'html.parser')

# Finde alle Artikel
articles = soup.find_all('li', class_='js-stream-content Pos(r)')

business_articles = []
# Überprüfe, ob das li ein div mit Inhalt "Business" hat und behalte nur diese
for article in articles:
    bsn = article.find_all('div', string='Business')
    if bsn:
        business_articles.append(article)

links = []
# Extrahiere den Link zum Artikel
for article in business_articles:
    link = article.find('a')
    # Füge finance.yahoo.com vorne an
    link = 'https://finance.yahoo.com' + link['href']
    links.append(link)

# Füge die Links in eine Datei ein
with open('links.txt', 'a') as f:
    for link in links:
        f.write(link + '\n')
```

Now that I had the links in a file, I could use the scraping script from before and just loop over all the links to scrape each articles text and save it to the database.

This was the first step of the creation process. Step two was to split the text into each sentence, save them to another table and then with the help of three BERT models ([FinancialBERT](https://huggingface.co/Sigma/financial-sentiment-analysis), [FinBERT](https://huggingface.co/ProsusAI/finbert) and [DistilRoBERTa](https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis)) classify each sentence into one of the following categories: "bullish", "bearish" or "neutral". Each model was trained on a different pre-trained BERT model and fine-tuned on the [financial_phrasebank](https://huggingface.co/datasets/financial_phrasebank) dataset.

For the sentence extraction part there exist libraries like NLTK but I found a little script that uses regular expressions and neat little tricks to do a better job:

```python
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"
multiple_dots = r'\.{2,}'

def remove_author_notes(text):
    # Broaden pattern to match updates or other notes at the beginning
    text = re.sub(r'^\([^\)]+\)\s*', '', text)

    # Pattern to match author and location, updated to handle variations
    text = re.sub(r'By\s.+?\s\(.+?\)\s*[-–]\s*', '', text)

    # Pattern to match Month Day () -
    text = re.sub(r'\w{3}\s\d{1,2}\s\([^)]*\)\s*-\s*', '', text)

    # Pattern to match additional reporting or editing notes at the end, updated for flexibility
    text = re.sub(r'\(Reporting by [\w\s,]+;\s*additional reporting by [\w\s,]+;\s*Editing by [\w\s,]+\)$', '', text, flags=re.IGNORECASE)

    return text

# https://stackoverflow.com/a/31505798 (source)
def split_into_sentences(text: str) -> list[str]:
    """
    Split the text into sentences.

    If the text contains substrings "<prd>" or "<stop>", they would lead
    to incorrect splitting because they are used as markers for splitting.

    :param text: text to be split into sentences
    :type text: str

    :return: list of sentences
    :rtype: list[str]
    """
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    if "No." in text: text = text.replace("No.","No<prd>")
    if "Jan." in text: text = text.replace("Jan.","Jan<prd>")
    if "Feb." in text: text = text.replace("Feb.","Feb<prd>")
    if "Mar." in text: text = text.replace("Mar.","Mar<prd>")
    if "Apr." in text: text = text.replace("Apr.","Apr<prd>")
    if "Jun." in text: text = text.replace("Jun.","Jun<prd>")
    if "Jul." in text: text = text.replace("Jul.","Jul<prd>")
    if "Aug." in text: text = text.replace("Aug.","Aug<prd>")
    if "Sep." in text: text = text.replace("Sep.","Sep<prd>")
    if "Sept." in text: text = text.replace("Sept.","Sept<prd>")
    if "Oct." in text: text = text.replace("Oct.","Oct<prd>")
    if "Nov." in text: text = text.replace("Nov.","Nov<prd>")
    if "Dec." in text: text = text.replace("Dec.","Dec<prd>")
    if "Corp." in text: text = text.replace("Corp.","Corp<prd>")
    if "Ltd." in text: text = text.replace("Ltd.","Ltd<prd>")
    if "vs." in text: text = text.replace("vs.","vs<prd>")
    if "e.g." in text: text = text.replace("e.g.","e<prd>g<prd>")
    if "i.e." in text: text = text.replace("i.e.","i<prd>e<prd>")
    if "Sen." in text: text = text.replace("Sen.","Sen<prd>")
    if "Calif." in text: text = text.replace("Calif.","Calif<prd>")
    if "Gov." in text: text = text.replace("Gov.","Gov<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    text = text.replace('<ellipsis>', '...')
    text = text.replace('<qst>', '?')
    text = text.replace('<exc>', '!')
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]: sentences = sentences[:-1]
    return sentences

def remove_parenthetical_sentences(sentences):
    cleaned_sentences = []
    for sentence in sentences:
        # Check if the entire sentence is within parentheses
        if sentence.startswith('(') and sentence.endswith(')'):
            continue  # Skip this sentence
        cleaned_sentences.append(sentence)
    return cleaned_sentences

def merge_parenthetical_statements(text):
    # Replace periods within parentheses with a placeholder
    text = re.sub(r'\(([^)]+)\)', lambda m: "(" + m.group(1).replace('.', '<prd>') + ")", text)
    return text

def replace_ellipses(text):
    # Replace '...' with a placeholder
    return re.sub(r'\.\.\.', '<ellipsis>', text)

def replace_sentence_stops_in_quotes(text):
    # Define a function to replace sentence stops within a matched quote
    def replace_stops(match):
        # Replace all periods, question marks, and exclamation marks in the matched quote
        temp = match.group(0).replace('.', '<prd>')
        temp = temp.replace('?', '<qst>')
        temp = temp.replace('!', '<exc>')
        return temp

    # Regex-Muster, das Text in Anführungszeichen erfasst
    quote_pattern = r'["“”](.*?)["“”]'

    # ersetzte alle Satzzeichen in Anführungszeichen
    text = re.sub(quote_pattern, replace_stops, text, flags=re.UNICODE)

    return text
```

With a little bit of tuning, this script was able to split the text into sentences the way I wanted. After the sentences have been split, they were saved into a table called **sentences** in the database.

The following script, gets all sentences, classifies them with the selected model and then saves the classification to the database:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# ProsusAI/finbert
# mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis
# Sigma/financial-sentiment-analysis
model_name = "ProsusAI/finbert"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# get each row from senteces table in sentiment.db sqlite
import sqlite3

conn = sqlite3.connect("data/sentiment.db")
c = conn.cursor()

c.execute("SELECT * FROM sentences WHERE finbert_result != 'bullish'")
rows = c.fetchall()

for row in rows:
    id = row[0]
    sentence_text = row[2]

    inputs = tokenizer(sentence_text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Wende das Modell an
    with torch.no_grad():
        outputs = model(**inputs)

    # Hole die Modellvorhersagen
    predictions = outputs.logits

    # Konvertiere Vorhersagen in Wahrscheinlichkeiten und hole das maximale
    predictions = torch.nn.functional.softmax(predictions, dim=1)
    predicted_class = torch.argmax(predictions).item()

    # behalte die Wahrscheinlichkeit für die Vorhersage
    predicted_probability = predictions.flatten()[predicted_class].item()

    # CLASSES ARE DIFFERENT FOR FINBERT
    classesFINBERT = ["bullish", "bearish", "neutral"]
    classes = ["bearish", "neutral", "bullish"]
    # print(f"Class: {classesFINBERT[predicted_class]}, Probability: {predicted_probability:.3f}")

    # update _result and _score column for each model but only take 3 decimals for _score
    c.execute("UPDATE sentences SET finbert_result = ?, finbert_score = ? WHERE id = ?", (classesFINBERT[predicted_class], round(predicted_probability, 3), id))
    if(id % 100 == 0):
        print(id)

conn.commit()
conn.close()
```

### IN PROGRESS
