import React, { useState, useEffect } from 'react';

interface GalleryProps {
	images: string[];
}

const Gallery: React.FC<GalleryProps> = ({ images }) => {
	const [currentImageIndex, setCurrentImageIndex] = useState(0);
	const [isModalOpen, setIsModalOpen] = useState(false);

	const handleNextImage = () => {
		setCurrentImageIndex((prevIndex) => (prevIndex + 1) % images.length);
	};

	const handlePrevImage = () => {
		setCurrentImageIndex((prevIndex) => (prevIndex - 1 + images.length) % images.length);
	};

	const openModal = () => {
		setIsModalOpen(true);
	};

	const closeModal = () => {
		setIsModalOpen(false);
	};

	useEffect(() => {
		let timer: NodeJS.Timeout;
		if (!isModalOpen) {
			timer = setInterval(() => {
				handleNextImage();
			}, 7000); // Change image every 3 seconds
		}

		// Clean up the timer when the component is unmounted or when the modal is opened
		return () => {
			if (timer) {
				clearInterval(timer);
			}
		};
	}, [isModalOpen]); // Re-run the effect when `isModalOpen` changes

	return (
		<div className="mt-4">
			<img
				key={currentImageIndex}
				src={images[currentImageIndex]}
				alt="Gallery Image"
				onClick={openModal}
				className="cursor-pointer rounded-md shadow-2xl hover:opacity-80"
			/>

			{isModalOpen && (
				<div className="fixed inset-0 z-10 flex items-center justify-center bg-black bg-opacity-40">
					{images.length > 1 && (
						<button onClick={handlePrevImage} className="ml-4">
							Previous
						</button>
					)}
					<div className="max-h-screen w-fit p-5 shadow-2xl">
						<img
							className="w-full"
							src={images[currentImageIndex]}
							alt="Gallery Image"
							onClick={closeModal}
						/>
					</div>
					{images.length > 1 && (
						<button onClick={handleNextImage} className="mr-4">
							Next
						</button>
					)}
				</div>
			)}
		</div>
	);
};

export default Gallery;
