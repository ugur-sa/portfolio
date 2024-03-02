import { defineCollection, z } from 'astro:content';

const projectCollection = defineCollection({
	schema: z.object({
		author: z.string(),
		title: z.string(),
		date: z.string(),
		image: z.string(),
		description: z.string()
	})
});

const postCollection = defineCollection({
	schema: z.object({
		title: z.string(),
		date: z.string(),
		author: z.string(),
		tags: z.string().array()
	})
});

export const collections = {
	projects: projectCollection,
	posts: postCollection
};
