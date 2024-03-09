---
title: How I created this portfolio
author: Ugur Sadiklar
date: 2024-03-06
tags: [markdown, astro, portfolio]
---

# Introduction

After working with Angular and React (Next.js), I wanted to explore new options for my portfolio. I chose to use Astro because it generates static sites and does not send any JavaScript to the client by default.

However, you may have noticed some client-side components, such as the gallery in the projects section, which are called islands in Astro. Islands are components that are rendered on the client-side and are not part of the static site. This is a useful feature because it allows the use of JavaScript only when necessary.

In this post, I will explain how I created this website.

# How I setup this project

I started by creating a new Astro project with the following command:

```bash
npm create astro@latest
```

Astro makes it extremely easy to integrate new functionality. I added Tailwind and React to the project with the following command:

```bash
npx astro add tailwind react
```

After that I was ready to go.

# How I created the gallery component

I wanted to display screenshots of my projects on their respective pages using React. To achieve this, you can create a `.jsx` or `.tsx` file in the components folder and imported it into the `.astro` file.

However, I encountered a problem while doing this - how to pass the images to the component and where to render them. To solve this, I passed the image paths in my markdown's frontmatter and then as a string array to the gallery component.

I mapped over the array and rendered the images. With a simple state variable, I stored the current image index and added buttons for image navigation.

However, the images were cut off due to the fixed size of the gallery. To improve the user experience, I added a modal that opens when the user clicks on an image. This modal displays the image in full size and allows the user to navigate through the images using the same buttons as in the gallery.

# IN PROGRESS
