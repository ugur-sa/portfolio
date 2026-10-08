---
title: 'Golf Athletik'
author: 'Ugur Sadiklar'
description: 'A CMS-driven website for Lukas Valdes, a TPI-certified golf fitness coach'
category:
  - astro
  - sanity
  - client work
date: 2026-10-08
created: 2026-06-12
image: /images/golfathletik/logo.svg
draft: false
gallery:
  - /images/golfathletik/home.png
  - /images/golfathletik/about.png
  - /images/golfathletik/methode.png
  - /images/golfathletik/screen.png
  - /images/golfathletik/leistungen.png
  - /images/golfathletik/ablauf.png
  - /images/golfathletik/termin.png
  - /images/golfathletik/faq.png
---

# Overview

Golf Athletik is the website for Lukas Valdes, a personal trainer and TPI-certified golf fitness coach in NRW, Germany. It is a one-page marketing site built around a single path: book a free intro call, then a TPI screening, then a training package or monthly plan. The main requirement was that Lukas can edit all the content himself, without needing a developer.

# Features

- One-page site with sections for about, method, services, process, booking and FAQ, designed for desktop and mobile
- Interactive 16-point TPI screen explorer (4×4 grid on desktop, swipeable cards on mobile)
- Booking through an embedded Cal.com calendar that only loads after cookie consent
- Embedded Sanity Studio at `/studio` with German labels, so Lukas can edit text, images, tests and FAQs himself
- SEO with structured data (LocalBusiness and FAQPage), a sitemap and self-hosted fonts
- Legal pages (Impressum, Datenschutz, AGB)

# Tech Stack

- Frontend: [Astro 5, React 19 (islands), Typescript, TailwindCSS]
- CMS: [Sanity]
- Hosting: [Vercel, with a Sanity webhook that rebuilds the site]
- Booking: [Cal.com]

# Challenges and Solutions

- **Content editing for a non-developer**: I set up Sanity Studio with a German, single-page-style sidebar and drag-and-drop ordering for tests and FAQs. When Lukas publishes a change, a webhook triggers a new Vercel build, so the site stays fully static and fast.

- **Not breaking when the CMS is unavailable**: All content is loaded with one GROQ query and merged over a local fallback file. If a field or document is missing, or the CMS can't be reached, the site still builds with the default content.

- **Turning a mockup into production**: The design came as standalone HTML mockups for desktop and mobile. I extracted the design tokens, fonts, logo and images from them and rebuilt everything as Astro components, using React only for the interactive parts.

# Results

The site is live at golfathletik.com, and all content can be updated through the CMS without touching the code.

# Lessons Learned

Besides learning Sanity and Astro 5, I learned how much the editing experience matters when someone else maintains the content, and how useful it is to plan for failures like a missing CMS.

# Further Information

- Live Project: [golfathletik.com](https://golfathletik.com)
- Source Code: private
