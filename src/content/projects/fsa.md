---
title: 'Sentiment Analysis'
author: 'Ugur Sadiklar'
description: 'A project to analyse sentiment of financial news articles'
category:
  - machine learning
  - bert
  - yahoo finance
date: 2024-03-05
image: /images/fsa/logo.png
draft: false
gallery:
  - /images/fsa/screenshot1.JPG
  - /images/fsa/screenshot2.JPG
---

# Overview

This project aimed to classify sentiment of financial news. Developed for my bachelor thesis, it showcases how financial news can impact market sentiment.

# Features

- Input Yahoo Finance News articles to be analysed
- See real-time status of the analysis
- View the sentiment of each sentence with a overall sentiment score

# Tech Stack

- Frontend: [Next.js, Typescript, TailwindCSS, shadcn/ui]
- Backend: [AWS Lambda, AWS API Gateway, Python, BERT (Huggingface)]
- Database: [SQLite, PostgreSQL]

# Challenges and Solutions

One of the main challenges faced during this project was creating my own dataset and training a pre-trained BERT model with it. Another challenge was to create a real-time analysis system, which was solved by using AWS Lambda and API Gateway.

# Results and Impact

The project was successfully launched in January 2024 and has been praised by my professors and peers.

# Lessons Learned

Throughout this project, I learned how to create datasets, train a BERT mode, and deploy a real-time analysis system.
I also had the opportunity to work with AWS Lambda and API Gateway, which was a great learning experience.

# Futher Information

- Live Project: [fsa.ugursadiklar.de](http://fsa.ugursadiklar.de)
- Source Code: [Github](https://github.com/ugur-sa/financial-sentiment-analysis)
- Blog Post: [Link](https://ugursadiklar.de/posts/howicreatedafinancialdatasetandtrainedbertonit)
- Bachelor Thesis: coming soon
