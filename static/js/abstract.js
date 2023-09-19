document.getElementById('generate').addEventListener('click', function () {
  var spanContainer = document.querySelector('.span-container');

  // Create a new span element
  var newSpan = document.createElement('span');

  // Fetch a new abstract
  fetch('/generate_abstract')
    .then(response => response.json())
    .then(section_texts => {
        // Get the container where you want to append the sections
        var container = document.getElementById('ajax-container');

        // Create a new span for all sections
        var allSectionsSpan = document.createElement('span');

        for (var section in section_texts) {
            // Create a new span for each section
            var sectionSpan = document.createElement('span');
            sectionSpan.className = `section ${section}`;
            sectionSpan.textContent = section_texts[section] + ' ';

            // Append the section span to the all sections span
            allSectionsSpan.appendChild(sectionSpan);
        }

        // Append the all sections span to the container
        container.appendChild(allSectionsSpan);
    });

  spanContainer.scrollTop = spanContainer.scrollHeight;
});
 