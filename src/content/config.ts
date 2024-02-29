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

export const collections = {
	projects: projectCollection
};
