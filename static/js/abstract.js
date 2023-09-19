document.getElementById('generate').addEventListener('click', function () {
  var spanContainer = document.querySelector('.span-container');

  for (var i = 0; i < 5; i++) {
    // Create a new span element
    var newSpan = document.createElement('span');

    // Fetch a new abstract
    fetch('/generate_abstract')
      .then(response => response.json())
      .then(section_texts => {
          // Get the container where you want to append the sections
          var container = document.getElementById('ajax-container');

          // Define the order of the sections
          var sectionOrder = ['Introduction', 'Method', 'Result', 'Discussion'];

          // Create a new span for all sections
          var allSectionsSpan = document.createElement('span');
          allSectionsSpan.className = 'abstract';

          for (var j = 0; j < sectionOrder.length; j++) {
              var section = sectionOrder[j];

              if (section in section_texts) {
                  // Create a new span for each section
                  var sectionSpan = document.createElement('span');
                  sectionSpan.className = `section ${section}`;
                  sectionSpan.textContent = section_texts[section] + ' ';

                  // Append the section span to the all sections span
                  allSectionsSpan.appendChild(sectionSpan);
              }
          }

          // Append the all sections span to the container
          container.appendChild(allSectionsSpan);

          // Add click event listener to the abstract
          allSectionsSpan.addEventListener('click', function() {
            // Remove 'selected' class from all abstracts
            var abstracts = document.querySelectorAll('.abstract');
            abstracts.forEach(function(abstract) {
              abstract.classList.remove('selected');
            });

            // Add 'selected' class to the clicked abstract
            this.classList.add('selected');
          });
      });

    spanContainer.scrollTop = spanContainer.scrollHeight;
  }
});

document.getElementById('submit').addEventListener('click', function () {
  // Get the selected abstract
  var selectedAbstract = document.querySelector('.abstract.selected');

  if (selectedAbstract) {
    // Get the text of the selected abstract
    var abstractText = selectedAbstract.textContent;

    // Send a POST request to the server-side route
    fetch('/update_abstract', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        abstract: abstractText
      })
    })
    .then(response => response.json())
    .then(data => {
      console.log(data);
    });
  } else {
    console.log('No abstract selected');
  }
});
