import { defineCollection, z } from 'astro:content';

const projectCollection = defineCollection({
	schema: z.object({
		author: z.string(),
		title: z.string(),
		date: z.date(),
		image: z.string().nullable(),
		description: z.string(),
		draft: z.boolean(),
		category: z.string().array(),
		gallery: z.string().array().optional()
	})
});

const postCollection = defineCollection({
	schema: z.object({
		title: z.string(),
		date: z.date(),
		author: z.string(),
		tags: z.string().array()
	})
});

export const collections = {
	projects: projectCollection,
	posts: postCollection
};
