document.getElementById('submit').addEventListener('click', function (event) {
    event.preventDefault();  // Prevent the form from being submitted normally

    const title = document.getElementById('title').value;
    const fileInput = document.getElementById('fileInput').files[0];

    if (!title || !fileInput) {
        alert('Please fill in all fields.');
        return;
    }

    if (fileInput.type !== 'application/vnd.openxmlformats-officedocument.wordprocessingml.document') {
        alert('Invalid file type. Please upload a .docx file.');
        return;
    }

    const formData = new FormData();
    formData.append('title', title);
    formData.append('file', fileInput);

    fetch('/upload', {
        method: 'POST',
        body: formData
    }).then(response => {
        if (response.ok) {
            alert('File uploaded successfully.');
            window.location.href = "/abstract"; // Redirect to abstract.html
        } else {
            alert('File upload failed.');
        }
    }).catch(error => {
        console.error('Error:', error);
    });
});
