var quill = new Quill('#editor', {
    modules: {
      toolbar: '#toolbar', 
    },
    theme: 'snow'
  });
  
  document.getElementById('generate-imrad').addEventListener('click', function() {
    fetch('/get_last_unapproved')
    .then(response => response.json())
    .then(data => {
        let file_path = data.file_path;
        fetch('/convert_docx_to_imrad', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({file_path: file_path})
        })
        .then(response => response.json())
        .then(data => {
            let converted_file_path = data.converted_file_path;
            fetch('/convert_docx_to_text', { 
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({file_path: converted_file_path})
            })
            .then(response => response.json())
            .then(data => {
                let text_content = data.text_content; 
      
                quill.setText(text_content);
            });
        });
    });
});

document.getElementById('submit').addEventListener('click', function() {
    window.location.href = '/abstract';
});
