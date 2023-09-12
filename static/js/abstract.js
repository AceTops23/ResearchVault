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
    
    // Create a new span element
    var newSpan = document.createElement('span');
    newSpan.classList.add('hidden');

    // Append the new span to the .span-container
    var spanContainer = document.querySelector('.span-container');
    spanContainer.appendChild(newSpan);

    // Scroll to the bottom to show the new span (optional)
    spanContainer.scrollTop = spanContainer.scrollHeight;

    // Make an AJAX request to the server
    fetch('/generate_abstract')
        .then(response => response.json())
        .then(section_texts => {
            // Display each section text in a new span within the container
            var container = document.querySelector('.container');
            container.innerHTML = '';  // Clear the container
            
            for (var section in section_texts) {
                var newSpan = document.createElement('span');
                newSpan.textContent = section + ': ' + section_texts[section];
                container.appendChild(newSpan);
            }
        });
});