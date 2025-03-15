document.addEventListener("DOMContentLoaded", function() {
    const figures = document.querySelectorAll('figure.img-figure');
    
    figures.forEach((figure, index) => {
      const caption = figure.querySelector('figcaption');
      if (caption) {
        caption.innerHTML = `<strong>Figure ${index + 1}:</strong> ${caption.innerHTML}`;
      }
    });
  });
  