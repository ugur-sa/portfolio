---
author: Ugur Sadiklar
category:
  - sim racing
  - formula1
date: 2024-03-05
created: 2023-02-13
featured: false
image: /images/raceboards/f1.webp
title: Raceboards
description: A simple project to keep track of lap times and races for sim racers
slug: raceboards
draft: false
gallery:
  - /images/raceboards/raceboards_bestTimes.png
  - /images/raceboards/TrackInformation.png
  - /images/raceboards/TrackInformation2.png
  - /images/raceboards/leaderboards.png
  - /images/raceboards/times.png
  - /images/raceboards/times2.png
  - /images/raceboards/progress.png
  - /images/raceboards/season2023.png
  - /images/raceboards/race.png
  - /images/raceboards/racegaps.png
  - /images/raceboards/qualilaps.png
---

# Overview

This is a project I built in early 2023 for my friends and I to use for sim racing. We were playing a game called _Assetto Corsa_ and I wanted to keep track of our lap times so we could compare and see who was the fastest on each track.

# Features

- Compare lap times on each Formula 1 track with your friends
- See medal leaderboard
- Add new lap times for each track
- Upload race results and review all sessions
- Upload races for a season to keep track of points
- See your own progress on the dashboard with a graph
- See track information and all recorded lap times in a table

# Tech Stack

- Frontend: [Next.js 12, Typescript, TailwindCSS]
- Backend: [Node.js, Prisma, Supabase (Auth)]
- Database: [PostgreSQL (Supabase)]

# Challenges and Solutions

At first it seemed like an impossible task to automate the process of adding race results. Luckily, Assetto Corsa creates a JSON file for each race with each session, lap times for each driver (with tires used, sectors, cuts, etc.). I wrote a parser to read these files and upload the data to the database so its easy to upload and review races.

Another challenge was to calculate the amount of laps a driver was in the lead. The only information I have is each drivers lap times, so what I did was to add up ever lap time for each driver and compare it to the other drivers each lap. The driver with the lowest time gets a point for that lap.
This seems like an easy task, but it was the most challenging part of this project.

# Results

The project is live and we are using it for our sim racing league and to compare our practice lap times for each Formula 1 track. It has been a great tool to keep track of our progress and to see who is the fastest on each track.

# Lessons Learned

I learned a lot about how to use Prisma and Supabase to create a simple backend for a project like this. As this was my first project after working for 5 months with Angular and ASP.NET at my internship, it was a great way to keep my skills sharp and learn new technologies.

# Futher Information

- Live Project: [raceboards.ugursadiklar.de](https://raceboards.ugursadiklar.de)
- Source Code: [Github](https://github.com/ugur-sa/raceboards)
