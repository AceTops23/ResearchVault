document.addEventListener("DOMContentLoaded", function () {
  const submitFormButton = document.querySelector("#submit-publish");
  const fileInput = document.querySelector("#fileInput");

  function submitForm(event) {
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
    if (file.type !== "application/pdf" && file.type !== "application/vnd.openxmlformats-officedocument.wordprocessingml.document") {
      alert("Please select a DOCX or PDF file.");
      return; // Stop form submission if the file is not a DOCX or PDF
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

    fetch("/submit_data", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.text())
      .then((data) => {
        // Handle any response data if needed
        alert("Upload successful!");
        window.location.href = "/"; // Redirect to abstract.html
      })
      .catch((error) => {
        // Handle any errors that occurred during the request
        console.error("Error:", error);
      });
  }

  submitFormButton.addEventListener("click", submitForm);
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

                                                   
document.getElementById('yearInput').addEventListener('change', updateYearInputValue);                                     