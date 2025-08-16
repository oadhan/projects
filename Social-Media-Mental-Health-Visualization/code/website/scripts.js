document.addEventListener("DOMContentLoaded", function () {
  const pages = document.querySelectorAll(".carousel-page");
  const prevBtn = document.getElementById("prevBtn");
  const nextBtn = document.getElementById("nextBtn");
  const pageLinks = document.querySelectorAll(".page-link-btn");
  let currentPage = 0;

  function showPage(index) {
    pages.forEach((page, i) => {
      page.classList.toggle("active", i === index);
    });
    pageLinks.forEach((btn, i) => {
      btn.classList.toggle("active", i === index);
    });
    currentPage = index;
  }

  prevBtn.addEventListener("click", () => {
    const newIndex = (currentPage - 1 + pages.length) % pages.length;
    showPage(newIndex);
  });

  nextBtn.addEventListener("click", () => {
    const newIndex = (currentPage + 1) % pages.length;
    showPage(newIndex);
  });

  pageLinks.forEach((btn) => {
    btn.addEventListener("click", (e) => {
      const targetPage = parseInt(e.target.getAttribute("data-page"), 10);
      if (!isNaN(targetPage)) showPage(targetPage);
    });
  });

  function initTableauVizzes() {
    const placeholders = document.querySelectorAll(".tableauPlaceholder");
    placeholders.forEach((container) => {
      const objectEl = container.querySelector("object.tableauViz");
      if (objectEl) {
        objectEl.style.display = "block";
        if (window.tableau && window.tableau.VizManager && window.tableau.VizManager.createViz) {
          window.tableau.VizManager.createViz(objectEl);
        }
      }
    });
  }

  setTimeout(initTableauVizzes, 1000);
  showPage(currentPage);
});
