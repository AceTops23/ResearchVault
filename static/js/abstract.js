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
});
