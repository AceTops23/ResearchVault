document.getElementById('generate').addEventListener('click', function() {
    // Get all the spans within the span-container
    var spans = document.querySelectorAll('.span-container span');
    
    // Find the first hidden span and unhide it
    for (var i = 0; i < spans.length; i++) {
        if (spans[i].classList.contains('hidden')) {
            spans[i].classList.remove('hidden');
            break; // Stop after unhiding the first hidden span
        }
    }
    
    // Scroll to the bottom to show the new span (optional)
    var spanContainer = document.querySelector('.span-container');
    spanContainer.scrollTop = spanContainer.scrollHeight;

    // Make an AJAX request to the server
    fetch('/generate_abstract')
    .then(response => response.json())
    .then(section_texts => {
        // Display each section text in a new div within the container
        var container = document.querySelector('#ajax-container');

        var newDiv = document.createElement('div');
        
        for (var section in section_texts) {
            newDiv.textContent += section + ': ' + section_texts[section] + ' ';
        }
        
        newDiv.innerHTML += '<br>';  // Add a line break after each abstract
        container.appendChild(newDiv);
    });
});
