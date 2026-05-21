// Copy BibTeX
function copyBibtex() {
  const text = document.getElementById('bibtex-block').textContent;
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.querySelector('.copy-btn');
    btn.innerHTML = '<i class="fas fa-check"></i> Copied!';
    btn.classList.add('copied');
    setTimeout(() => {
      btn.innerHTML = '<i class="fas fa-copy"></i> Copy';
      btn.classList.remove('copied');
    }, 2000);
  });
}

// Smooth active nav highlighting (optional enhancement)
document.addEventListener('DOMContentLoaded', () => {
  // If paper / code links are placeholder, make them open gracefully
  const paperLink = document.getElementById('paper-link');
  const codeLink = document.getElementById('code-link');

  if (paperLink && paperLink.getAttribute('href') === '#') {
    paperLink.addEventListener('click', e => {
      e.preventDefault();
      alert('Paper PDF link coming soon.');
    });
  }
  if (codeLink && codeLink.getAttribute('href') === '#') {
    codeLink.addEventListener('click', e => {
      e.preventDefault();
      alert('Code repository coming soon.');
    });
  }
});
