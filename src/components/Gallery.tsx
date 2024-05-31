import React, { useState } from 'react';
import { MdChevronLeft, MdChevronRight } from 'react-icons/md';
import { RxDotFilled } from 'react-icons/rx';
import { IoClose } from 'react-icons/io5';

interface GalleryProps {
	images: string[];
}

const Gallery: React.FC<GalleryProps> = ({ images }) => {
	const [currentImage, setCurrentImage] = useState(0);
	const [isModalOpen, setIsModalOpen] = useState(false);

	const openModal = () => {
		setIsModalOpen(true);
		//disable scrolling
		document.body.style.overflow = 'hidden';
	};

	const closeModal = () => {
		setIsModalOpen(false);
		//enable scrolling
		document.body.style.overflow = 'auto';
	};

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
		<div className="group m-auto h-[450px] w-full rounded-lg border border-zinc-600 md:max-h-[500px] lg:max-h-[500px]">
			<div className="relative h-full overflow-hidden">
				<div
					className="absolute -z-50 h-full w-full opacity-60 blur-xl"
					style={{
						backgroundImage: `url(${images[currentImage]})`
					}}
				/>
				<div
					className="h-full w-full cursor-zoom-in rounded-2xl bg-contain bg-center bg-no-repeat duration-500"
					style={{ backgroundImage: `url(${images[currentImage]})` }}
					onClick={openModal}
				></div>
				{/* Left Arrow*/}
				<div className="absolute left-5 top-[50%] hidden -translate-x-0 translate-y-[-50%] cursor-pointer rounded-full bg-black/20 p-1 text-2xl text-white hover:bg-black/50 group-hover:block">
					<MdChevronLeft size={30} onClick={previousSlide} />
				</div>
				{/* Right Arrow*/}
				<div className="absolute right-5 top-[50%] hidden -translate-x-0 translate-y-[-50%] cursor-pointer rounded-full bg-black/20 p-1 text-2xl text-white hover:bg-black/50 group-hover:block">
					<MdChevronRight size={30} onClick={nextSlide} />
				</div>
			</div>
			<div className="m-auto mt-2 flex w-fit rounded-full bg-white/20">
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
			<div className="hidden xl:block">
				{isModalOpen && (
					<div className="fixed left-0 top-0 z-50 flex h-full w-full items-center justify-center bg-black bg-opacity-50">
						<div className="relative flex h-full w-full flex-col items-center justify-center">
							<div
								style={{
									backgroundImage: `linear-gradient(rgba(0, 0, 0, 0.8), rgba(0, 0, 0, 0.8)), url(${images[currentImage]})`
								}}
								className="absolute inset-0 -z-50 -m-4 bg-cover bg-center bg-no-repeat p-4 blur-xl filter"
							/>
							<button
								onClick={closeModal}
								className="absolute right-0 top-0 z-50 mr-2 mt-2 rounded-full bg-black/30 p-2"
							>
								<IoClose size={30} />
							</button>
							<img
								src={images[currentImage]}
								alt="Modal"
								style={{ objectFit: 'contain', maxHeight: '100vh', maxWidth: '100vw' }}
							/>

							{/* Left Arrow*/}
							<div className="absolute left-5 top-[50%] hidden -translate-x-0 translate-y-[-50%] cursor-pointer rounded-full bg-black/20 p-1 text-2xl text-white group-hover:block">
								<MdChevronLeft size={30} onClick={previousSlide} />
							</div>
							{/* Right Arrow*/}
							<div className="absolute right-5 top-[50%] hidden -translate-x-0 translate-y-[-50%] cursor-pointer rounded-full bg-black/20 p-1 text-2xl text-white group-hover:block">
								<MdChevronRight size={30} onClick={nextSlide} />
							</div>
							<div className="absolute bottom-10 flex justify-center rounded-full bg-black/20">
								{images.map((slide, slideIndex) => (
									<div
										key={slideIndex}
										className="cursor-pointer text-lg"
										onClick={() => goToSlide(slideIndex)}
									>
										<RxDotFilled
											className={slideIndex === currentImage ? 'text-white' : 'text-gray-500'}
										/>
									</div>
								))}
							</div>
						</div>
					</div>
				)}
			</div>
		</div>
	);
};

export default Gallery;
