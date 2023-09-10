// Add an event listener to the 'sort' select element
document.getElementById('sort').addEventListener('change', function () {
    filterItems();
});

// Add an event listener to the 'year-input' input element
document.getElementById('year-input').addEventListener('input', function () {
    filterItems();
});

document.getElementById('search-button').addEventListener('click', function () {
  performSearch();
});

// Function to handle search and filtering
function handleSearchAndFilter() {
  const searchQuery = document.getElementById('search-input').value;
  const selectedField = document.getElementById('field').value;
  const selectedSort = document.getElementById('sort').value;
  const selectedYear = document.getElementById('year-input').value;

  let url = '/browse?';
  if (searchQuery) {
      url += `search=${encodeURIComponent(searchQuery)}&`;
  }
  if (selectedField) {
      url += `field=${encodeURIComponent(selectedField)}&`;
  }
  if (selectedSort) {
      url += `sort=${encodeURIComponent(selectedSort)}&`;
  }
  if (selectedYear) {
      url += `year=${encodeURIComponent(selectedYear)}&`;
  }

  window.location.href = url;
}

// Add event listeners to the search button and filter elements
document.getElementById('search-button').addEventListener('click', handleSearchAndFilter);

document.getElementById('field').addEventListener('change', handleSearchAndFilter);

document.getElementById('sort').addEventListener('change', handleSearchAndFilter);

document.getElementById('year-input').addEventListener('change', handleSearchAndFilter);

document.getElementById('search-form').addEventListener('submit', function (event) {
  event.preventDefault(); // Prevent form submission
  handleSearchAndFilter();
});

document.getElementById('search-input').addEventListener('input', function () {
  // Prevent automatic updates when typing in the search bar
});

  

document.addEventListener('DOMContentLoaded', function () {
    // Get the search input element
    const searchInput = document.getElementById('search-input');
  
    // Add a 'keyup' event listener to the search input
    searchInput.addEventListener('keyup', function () {
      // Get the search query
      const searchQuery = searchInput.value.trim().toLowerCase();
  
      // Get the selected field option
      const selectedField = document.getElementById('field').value;
  
      // Get the selected sort option
      const selectedSort = document.getElementById('sort').value;
  
      // Get the selected year option
      const selectedYear = document.getElementById('year-input').value;
  
      // Get all the browse items
      const browseItems = document.getElementsByClassName('browse-item');
  
      // Loop through the browse items and check if each item matches the search query and selected filter options
      for (const item of browseItems) {
        const subjectArea = item.dataset.subjectarea.toLowerCase();
        const title = item.textContent.toLowerCase();
  
        const fieldMatch = !selectedField || subjectArea === selectedField.toLowerCase();
        const yearMatch = !selectedYear || item.dataset.year === selectedYear;
        const searchMatch = !searchQuery || title.includes(searchQuery);
  
        // Determine whether to show or hide the item based on the matches
        if (fieldMatch && yearMatch && searchMatch) {
          item.style.display = 'block';
        } else {
          item.style.display = 'none';
        }
      }
    });
  });
  
  