document.getElementById('generate').addEventListener('click', function () {
    var spans = document.querySelectorAll('.span-container span');
  
    for (var i = 0; i < spans.length; i++) {
      if (spans[i].classList.contains('hidden')) {
        spans[i].classList.remove('hidden');
  
        fetch('/generate_abstract')
          .then(response => response.json())
          .then(section_texts => {
            var newSpan = spans[i];
     
            var abstractText = '';
            for (var section in section_texts) {
              abstractText += section + ': ' + section_texts[section] + ' ';
            }
  
            newSpan.textContent = abstractText;
  
            var lineBreak = document.createElement('br');
            newSpan.appendChild(lineBreak);
          });
  
        break; 
      }
    }
  
    var newSpan = document.createElement('span');
    newSpan.classList.add('hidden');
    var spanContainer = document.querySelector('.span-container');
    spanContainer.appendChild(newSpan);
  
    spanContainer.scrollTop = spanContainer.scrollHeight;
  });
  