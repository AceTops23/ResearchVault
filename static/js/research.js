document.getElementById('search-button').addEventListener('click', function () {
  performSearch();
});

function performSearch() {
  const searchQuery = document.getElementById('search-input').value;

  let url = '/research?';
  if (searchQuery) {
      url += `search=${encodeURIComponent(searchQuery)}&`;
  }

  window.location.href = url;
}

document.addEventListener('DOMContentLoaded', function () {
    const searchInput = document.getElementById('search-input');
  
    searchInput.addEventListener('keyup', function () {
      const searchQuery = searchInput.value.trim().toLowerCase();
      const browseItems = document.getElementsByClassName('browse-item');
  
      for (const item of browseItems) {
        const title = item.textContent.toLowerCase();
        const searchMatch = !searchQuery || title.includes(searchQuery);
  
        if (searchMatch) {
          item.style.display = 'block';
        } else {
          item.style.display = 'none';
        }
      }
    });
});
