---
title: 'StackTracker'
author: 'Ugur Sadiklar'
description: 'A privacy-first web app for tracking precious-metal investments with live spot prices'
category:
  - next.js
  - supabase
  - precious metals
date: 2026-10-08
created: 2026-06-26
image: /images/stacktracker/logo.png
draft: false
gallery:
  - /images/stacktracker/dashboard.png
  - /images/stacktracker/holdings.png
  - /images/stacktracker/coin-detail.png
  - /images/stacktracker/lot-detail.png
  - /images/stacktracker/sell-lot.png
  - /images/stacktracker/gold-chart.png
  - /images/stacktracker/spot-prices.png
  - /images/stacktracker/alert.png
---

# Overview

StackTracker is a web app for tracking precious-metal investments (gold, silver, platinum and palladium) across multiple portfolios. It shows live spot prices, profit and loss down to each individual purchase, and realistic buy/sell estimates based on dealer premiums. It works without an account: everything can stay in the browser and later be migrated to a cloud account.

# Features

- Dashboard with total portfolio value, interactive value chart and metal breakdown
- Holdings grouped into positions, expandable to individual lots with per-lot P&L, serials and documents
- Live spot prices with long-term price charts going back to 1915
- Multiple portfolios, including parent portfolios that roll up their children
- Import wizard for CSV/XLSX/JSON exports, parsed and matched against the coin catalog entirely in the browser
- Public coin and bar catalog
- Local mode without an account, with migration to a cloud account on sign-up
- XLSX/PDF inventory export
- English, German and Turkish translations
- Free and Premium plans
- Admin console for managing the catalog, mints, prices and feature flags

# Tech Stack

- Frontend: [Next.js 16, React 19, Typescript, TailwindCSS v4, Radix UI]
- Backend: [Next.js Route Handlers, Supabase (Auth, Storage), Stripe]
- Database: [PostgreSQL (Supabase) with Row Level Security]
- Testing: [Vitest]

# Challenges and Solutions

- **Reliable spot prices**: The app gets live prices from a free feed without an API key. If that feed goes down, it switches to a paid backup source, and if that fails too, it uses fixed fallback prices so the app never breaks. A circuit breaker checks the main feed every 5 minutes and switches back once it recovers.

- **Working without an account**: Local mode stores portfolios in `localStorage` behind a fake Supabase client, so the same `supabase.from(...)` calls work in both modes. When a user signs up, all local data is migrated into the new account.

- **Private imports**: The import wizard does all its work in the browser: parsing, auto-detecting columns (with presets for common exports), matching rows to the catalog and reviewing drafts. Nothing gets uploaded and no account is needed.

- **Correct valuations**: All valuation logic is written as pure, unit-tested functions that convert everything to price per gram, so troy ounces, fineness, premiums and EUR/USD conversion are handled consistently everywhere.

# Results

StackTracker is live and in active development, with over 500 commits so far.

# Lessons Learned

I learned a lot about designing a data model with Row Level Security, keeping a large codebase maintainable with pure, tested domain logic, building a fully translated app, and designing for privacy from the start.

# Further Information

- Live Project: [stacktracker.app](https://stacktracker.app)
- Source Code: private
