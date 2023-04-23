
/*______________________STAR RATING________________________*/
// Get all star rating containers
const starRatingContainers = document.querySelectorAll('.star-rating');

// Add event listeners to each container
starRatingContainers.forEach((container) => {
  const stars = container.querySelectorAll('.fa-star');
  const resetButton = container.querySelector('.btn');
  const result = container.querySelector('.result');
  let rating = 0;
  
  const handleStarHover = (e) => {
    const starIndex = parseInt(e.target.dataset.index);
    highlightStars(starIndex);
  };
  
  const resetRating = () => {
    rating = 0;
    highlightStars(0);
    result.textContent = '';
  };
  
  const highlightStars = (starIndex) => {
    stars.forEach((star, index) => {
      if (index < starIndex) {
        star.classList.add('checked');
      } else {
        star.classList.remove('checked');
      }
    });
  };
  
  stars.forEach((star, index) => {
    star.addEventListener('mouseover', handleStarHover);
    star.addEventListener('click', () => {
      rating = index + 1;
    //   result.textContent = `You rated this ${rating} stars!`;
    });
    star.dataset.index = index + 1;
  });
  
  resetButton.addEventListener('click', resetRating);
});

/*______________________GO TOP BTN________________________*/

let scrollToTopBtn = document.querySelector(".scrollToTopBtn");
let rootElement = document.documentElement;

function handleScroll() {
  // Do something on scroll
  let scrollTotal = rootElement.scrollHeight - rootElement.clientHeight;
  if (rootElement.scrollTop / scrollTotal > 0.2) {
    // Show button
    scrollToTopBtn.classList.add("showBtn");
  } else {
    // Hide button
    scrollToTopBtn.classList.remove("showBtn");
  }
}

function scrollToTop() {
  // Scroll to top logic
  rootElement.scrollTo({
    top: 0,
    behavior: "smooth"
  });
}
scrollToTopBtn.addEventListener("click", scrollToTop);
document.addEventListener("scroll", handleScroll);





