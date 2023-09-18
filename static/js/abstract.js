document.getElementById('generate').addEventListener('click', function () {
  var spanContainer = document.querySelector('.span-container');

  // Create a new span element
  var newSpan = document.createElement('span');

  // Fetch a new abstract
  fetch('/generate_abstract')
      .then(response => response.json())
      .then(section_texts => {
          var abstractText = '';
          for (var section in section_texts) {
              abstractText += section_texts[section] + ' ';
          }

          // Update the text content of the new span element
          newSpan.textContent = abstractText;

          var lineBreak = document.createElement('br');
          newSpan.appendChild(lineBreak);
      });

  // Append the new span element to the span container
  spanContainer.appendChild(newSpan);

  spanContainer.scrollTop = spanContainer.scrollHeight;
});
 