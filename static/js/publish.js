document.addEventListener("DOMContentLoaded", function () {
  const approveForm = document.querySelector("#approve-form");
  const unapproveForm = document.querySelector('#unapprove-form');
  const fileInput = document.querySelector("#fileInput");

  function submitForm(event, status) {
    event.preventDefault();

    const title = document.querySelector("#title").value;
    const authors = document.querySelector("#authors").value;
    const publicationDate = document.querySelector("#yearInput").value;
    const thesisAdvisor = document.querySelector("#thesisAdvisor").value;
    const department = document.querySelector("#department").value;
    const degree = document.querySelector("#degree").value;
    const subjectArea = document.querySelector("#subjectArea").value;
    const abstract = document.querySelector("#abstract").value;
    const file = fileInput.files[0];

    // Check if any required field is empty
    if (!title || !authors || !publicationDate || !thesisAdvisor || !department || !degree || !subjectArea || !abstract || !file) {
      alert("Please fill in all fields and select a file.");
      return; // Stop form submission if any field is missing
    }

    // Check if the selected file is a DOCX or PDF
    if (status === "Published" && file.type !== "application/pdf") {
      alert("Please select a PDF file.");
      return; // Stop form submission if the file is not a PDF
    } else if (status === "Working" && file.type !== "application/vnd.openxmlformats-officedocument.wordprocessingml.document") {
      alert("Please select a DOCX file.");
      return; // Stop form submission if the file is not a DOCX
    }

    const formData = new FormData();
    formData.append("title", title);
    formData.append("authors", authors);
    formData.append("publicationDate", publicationDate);
    formData.append("thesisAdvisor", thesisAdvisor);
    formData.append("department", department);
    formData.append("degree", degree);
    formData.append("subjectArea", subjectArea);
    formData.append("abstract", abstract);
    formData.append("file", file);
    
    // Add status to form data
    formData.append("status", status);

    fetch("/submit_data", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.text())
      .then((data) => {
        // Handle any response data if needed
        alert("Upload successful!");
        window.location.href = "/abstract"; // Redirect to index.html
      })
      .catch((error) => {
        // Handle any errors that occurred during the request
        console.error("Error:", error);
      });
  }

  unapproveForm.addEventListener("submit", function (event) {
    submitForm(event, "Working");
  });

  approveForm.addEventListener("submit", function (event) {
    submitForm(event, "Published");
  });
});


$(document).ready(function() {
  // Check session state on page load
  $.getJSON("/session_state", function(data) {
    if (!data.isLoggedIn) {
      window.location.href = "/"; // Redirect to index.html
    }
  });
});

// Function to update the year input value without the month
function updateYearInputValue() {
  const monthInput = document.getElementById('yearInput');
  const year = monthInput.value.slice(0, 4); // Extract the year from the input value
  monthInput.value = year;
}

// Attach the updateYearInputValue function to the change event of the month input
  document.getElementById('yearInput').addEventListener('change', updateYearInputValue);


  const submitButton = document.getElementById('submit-publish');
  const pocContainer = document.getElementById('poc-container');
  
  submitButton.addEventListener('click', function(event) {
    console.log('clicked');
    pocContainer.classList.toggle('hidden');
  });

                                                     