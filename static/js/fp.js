// Find the cancel button by its ID
const cancelButton = document.getElementById('fp-cancel');

// Add a click event listener to the cancel button
cancelButton.addEventListener('click', function() {
  // Redirect the user to the root URL
  window.location.href = '/';
});


//login//
// Event listener for login functionality
document.getElementById('fp-login').addEventListener('click', function() {
  // Get the email and password input values
  const email = document.querySelector('#fp-email').value;
  const password = document.querySelector('#fp-password').value;

  // Perform login validation using SQL
  fetch('validate_login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email: email, password: password })
  })
      .then(response => {
          if (response.ok) {
              return response.json();
          } else {
              throw new Error('Login request failed.');
          }
      })
      .then(data => {
          const response = data;
          if (response.success) {
              alert('Login successful!');
              // Perform any necessary actions after successful login
              window.location.href = '/'; // Change the URL to your desired index page
          } else {
              if (!response.emailExists && !response.passwordMatch) {
                  alert('Invalid email and password. Please try again.');
              } else if (!response.emailExists) {
                  alert('Invalid email. Please try again.');
              } else if (!response.passwordMatch) {
                  alert('Incorrect password. Please try again.');
              }
          }
      })
});