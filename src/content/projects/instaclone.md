---
author: 'Ugur Sadiklar'
category:
  - clone
  - next.js
date: 2024-03-05
created: 2023-06-29
featured: false
title: 'Instaclone'
description: 'An Instagram clone built with Next.js'
draft: false
gallery:
  - /images/instaclone/home.png
  - /images/instaclone/post.png
  - /images/instaclone/upload1.png
  - /images/instaclone/upload2.png
  - /images/instaclone/upload3.png
  - /images/instaclone/profile.png
  - /images/instaclone/followers.png
---

# Overview

This project is an Instagram clone I built to learn about Infinite Scrolling, Image Uploading, Websockets for real-time chat and more. It's a simple project that is implemented with Next.js and deployed to Vercel.

# Features

- Infinite Scrolling feed
- Create Posts (image upload, caption)
- Comment on posts
- Like/Unlike posts and comments
- Follow/Unfollow users
- Real-time chat (in development)

# Tech Stack

- Frontend: [Next.js, Typescript, TailwindCSS]
- Backend: [Node.js, Prisma, Uploadthing, Websockets (Socket.io), Clerk (Auth)]
- Database: [MySQL (Planetscale)]

# Challenges and Solutions

- **Infinite Scrolling**: I had to implement infinite scrolling for the feed. I used the Intersection Observer API to detect when the user reaches the bottom of the page and fetch more posts.

- **Image Uploading**: I had to implement image uploading for posts. I used Uploadthing to handle file uploads and store the images in an S3 bucket.

- **Real-time chat**: I had to implement real-time chat for users. I used Websockets (Socket.io) to create a real-time chat feature.

# Lessons Learned

This project was a great way to learn a lot about Next.js and especially how routing works. I also had the opportunity to learn Websockets to create a real-time chat feature and how to handle file uploads.

# Results

I was able to build some of the core features of Instagram like creating posts, commenting, liking, following users and real-time chat. Even though the application needs a lot of optimization and improvements, I had a lot of fun building it and learned a lot in the process.

# Further Information

- Live Project: not deployed
- Source Code: [Github](https://github.com/ugur-sa/instaclone)
