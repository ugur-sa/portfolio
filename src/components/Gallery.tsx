import React, { useState } from 'react';
import { BsChevronCompactLeft, BsChevronCompactRight } from 'react-icons/bs';
import { RxDotFilled } from 'react-icons/rx';

interface GalleryProps {
	images: string[];
}

const Gallery: React.FC<GalleryProps> = ({ images }) => {
	const [currentImage, setCurrentImage] = useState(0);

	const previousSlide = () => {
		setCurrentImage((prev) => (prev === 0 ? images.length - 1 : prev - 1));
	};
	const nextSlide = () => {
		setCurrentImage((prev) => (prev === images.length - 1 ? 0 : prev + 1));
	};

	const goToSlide = (index: number) => {
		setCurrentImage(index);
	};

	return (
		<div className="group relative m-auto h-[300px] w-full py-4 md:h-[400px] lg:h-[500px]">
			<div
				className="h-full w-full rounded-2xl bg-cover bg-center duration-500"
				style={{ backgroundImage: `url(${images[currentImage]})` }}
			></div>
			{/* Left Arrow*/}
			<div className="absolute left-5 top-[50%] hidden -translate-x-0 translate-y-[-50%] cursor-pointer rounded-full bg-black/20 p-2 text-2xl text-white hover:bg-black/50 group-hover:block">
				<BsChevronCompactLeft size={30} onClick={previousSlide} />
			</div>
			{/* Right Arrow*/}
			<div className="absolute right-5 top-[50%] hidden -translate-x-0 translate-y-[-50%] cursor-pointer rounded-full bg-black/20 p-2 text-2xl text-white hover:bg-black/50 group-hover:block">
				<BsChevronCompactRight size={30} onClick={nextSlide} />
			</div>
			<div className="top-4 flex justify-center py-2">
				{images.map((slide, slideIndex) => (
					<div
						key={slideIndex}
						className="cursor-pointer text-lg"
						onClick={() => goToSlide(slideIndex)}
					>
						<RxDotFilled className={slideIndex === currentImage ? 'text-white' : 'text-gray-500'} />
					</div>
				))}
			</div>
		</div>
	);
};

export default Gallery;
