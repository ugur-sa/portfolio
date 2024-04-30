---
title: How I Created This Portfolio
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

# Home page

To begin with, I started creating the home page. I took inspiration from [Lee Robinson's portfolio](https://leerob.io/) and created a similar layout. I used Tailwind CSS to style the page and added a few animations to make it more dynamic.

The home page is supposed to give an overview about what I do right now and what projects I have worked on.
For that I created a grid layout to display my four most recent projects.

I also added a table to show my three most recent blog posts. The blog posts are a way for me to consolidate what I have learned and maybe share that knowledge with others.

# Experience page

This page was created to show my technical skills like programming languages, frameworks, and tools I have worked with. Additionally, I wanted to show
my work experience and education. For the skills section I created a 'SkillCard' component that displays the skill with a logo. All these SkillCards are then combined in a grid layout and displayed to the user.

For the work experience and education I created a 'ExperienceCard' component that takes in company/school name, position, start and end date etc. and creates a header and footer for each card. These cards then have children props that are displayed in the body of the card. In here I can add a list of tasks I did at that job or the courses I took at that school. This provides a nice overview of my resume.

# Projects page

The projects page is supposed to show the user what projects I have worked on sorted by the most recent. For that I display the projects in a Card component that shows the project name, description, and a button to view the project. When the user clicks on the button, they are redirected to the project page. Here the user can see older projects that are not displayed on the home page.

# Blog page

The blog page is supposed to show the user all the blog posts I have written. I created a Card component that displays the blog post title, date, and a button to read the post. When the user clicks on the button, they are redirected to the blog post page. The blog is a way for me to consolidate what I have learned and if possible share that knowledge with others.

# Conclusion

I am very happy with the result of this project. I have learned a lot about Astro and how to create a portfolio with it. I am excited to see what I can create with Astro in the future.

<!-- # How I created the gallery component

I wanted to display screenshots of my projects on their respective pages using React. To achieve this, you can create a `.jsx` or `.tsx` file in the components folder and imported it into the `.astro` file.

However, I encountered a problem while doing this - how to pass the images to the component and where to render them. To solve this, I passed the image paths in my markdown's frontmatter and then as a string array to the gallery component.

I mapped over the array and rendered the images. With a simple state variable, I stored the current image index and added buttons for image navigation.

However, the images were cut off due to the fixed size of the gallery. To improve the user experience, I added a modal that opens when the user clicks on an image. This modal displays the image in full size and allows the user to navigate through the images using the same buttons as in the gallery. -->
