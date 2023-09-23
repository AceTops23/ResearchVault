const loginButton = document.getElementById('login-button');
const loginContainer = document.getElementById('login-container');
const signupButton = document.getElementById('signup-button');
const signupContainer = document.getElementById('signup-container');
const loginButtonDropdown = document.getElementById('login-button-dropdown');
const signupButtonDropdown = document.getElementById('signup-button-dropdown');
const emailInput = document.getElementById('email-signup');
const passwordInput = document.getElementById('password-signup');


document.addEventListener('DOMContentLoaded', function() {
  var menuIcon = document.getElementById('menu-icon');
  var rightDropdown = document.querySelector('.right-dropdown');

  menuIcon.addEventListener('click', function() {
      if (rightDropdown.style.visibility === 'hidden') {
          rightDropdown.style.visibility = 'visible'; // Show the dropdown
      } else {
          rightDropdown.style.visibility = 'hidden'; // Hide the dropdown
          signupContainer.classList.add('hidden');
          signupContainer.classList.remove('visible');
          loginContainer.classList.add('hidden');
          loginContainer.classList.remove('visible');
      }
      
  });
});

document.addEventListener('DOMContentLoaded', function() {
  var closeLogin = document.getElementById('close-login');
  var loginContainer = document.getElementById('login-container');
  var loginButton = document.getElementById('login-button');

  closeLogin.addEventListener('click', function() {
      loginContainer.classList.add('hidden'); // Hide the login container
  });

  loginButton.addEventListener('click', function() {
      loginContainer.classList.remove('hidden'); // Show the login container
  });
});

document.addEventListener('DOMContentLoaded', function() {
    var closeSignup = document.getElementById('close-signup');
    var signupContainer = document.getElementById('signup-container');
    var signupButton = document.getElementById('signup-button');

    closeSignup.addEventListener('click', function() {
        signupContainer.classList.add('hidden'); // Hide the signup container
    });

    signupButton.addEventListener('click', function() {
        signupContainer.classList.remove('hidden'); // Show the signup container
    });
});

loginButtonDropdown.addEventListener('click', function() {
  loginContainer.classList.toggle('hidden');
  loginContainer.classList.toggle('visible');
  signupContainer.classList.add('hidden');
  signupContainer.classList.remove('visible');
});

signupButtonDropdown.addEventListener('click', function() {
  signupContainer.classList.toggle('hidden');
  signupContainer.classList.toggle('visible');
  loginContainer.classList.add('hidden');
  loginContainer.classList.remove('visible');
});

loginButton.addEventListener('click', function() {
  loginContainer.classList.toggle('hidden');
  loginContainer.classList.toggle('visible');
  signupContainer.classList.add('hidden');
  signupContainer.classList.remove('visible');
});

signupButton.addEventListener('click', function() {
  signupContainer.classList.toggle('hidden');
  signupContainer.classList.toggle('visible');
  loginContainer.classList.add('hidden');
  loginContainer.classList.remove('visible');
});

//login//
// Event listener for login functionality
document.getElementById('login').addEventListener('click', function() {
  // Get the email and password input values
  const email = document.querySelector('#email-login').value;
  const password = document.querySelector('#password-login').value;

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
              updateButtons(response.isLoggedIn);
              document.getElementById('login-container').classList.add('hidden');
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
      .catch(error => {
          console.error('Login request failed:', error);
          alert('An error occurred during login validation. Please try again later.');
      });
});


// Sign up
// Event listener for creating an account
document.querySelector('.get-started').addEventListener('click', function() {
  // Get input field values
  const lastName = document.querySelector('#signup-container input[type="text"][placeholder="Last Name"]').value;
  const firstName = document.querySelector('#signup-container input[type="text"][placeholder="First Name"]').value;
  const email = document.querySelector('#email-signup').value;
  const password = document.querySelector('#password-signup').value;

  // Check if required fields are not empty
  if (lastName.trim() === '' || firstName.trim() === '' || email.trim() === '' || password.trim() === '') {
      alert('Please fill in all the required fields.');
      return;
  }

  // Perform account creation using SQL
  fetch('create_account', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ lname: lastName, fname: firstName, email: email, password: password })
  })
      .then(response => {
          if (response.ok) {
              return response.json();
          } else {
              throw new Error('Account creation request failed.');
          }
      })
      .then(data => {
          const response = data;
          if (response.success) {
              alert('Account created successfully!');
              // Perform any necessary actions after successful account creation
          } else {
              if (response.emailExists) {
                  alert('The provided email already exists. Please use a different email.');
              } else {
                  alert('Account creation failed. Please try again.');
              }
          }
      })
      .catch(error => {
          console.error('Account creation request failed:', error);
          alert('An error occurred during account creation. Please try again later.');
      });
});

// Update login and signup buttons based on session state and update button behavior based on login status
function updateButtons(isLoggedIn) {
  const loginButton = document.getElementById('login-button');
  const signupButton = document.getElementById('signup-button');
  const loginButtonDD = document.getElementById('login-button-dropdown');
  const signupButtonDD = document.getElementById('signup-button-dropdown');
  const logoutButtonDD = document.getElementById('logout-button-dropdown');
  const logoutButton = document.createElement('button');
  logoutButton.id = 'logout-button';
  logoutButton.innerText = 'LOGOUT';

  if (isLoggedIn) {
    loginButton.style.display = 'none';
    signupButton.style.display = 'none';
    loginButtonDD.style.display = 'none';
    signupButtonDD.style.display = 'none';
    logoutButtonDD.style.display = 'block';
    
    document.querySelector('.right').appendChild(logoutButton);
    startButton.addEventListener('click', toggleContainer);
  } else {
    loginButton.style.display = 'block';
    signupButton.style.display = 'block';
    loginButtonDD.style.display = 'block';
    signupButtonDD.style.display = 'block';
    logoutButtonDD.style.display = 'none';
    const existingLogoutButton = document.getElementById('logout-button');
    if (existingLogoutButton) {
      existingLogoutButton.remove();
    }
    startButton.removeEventListener('click', toggleContainer);
  }
}

// Get session state
function getSessionState() {
  fetch('/session_state')
    .then(response => {
      if (response.ok) {
        return response.json();
      } else {
        throw new Error('Failed to retrieve session state.');
      }
    })
    .then(data => {
      const isLoggedIn = data.isLoggedIn;
      updateButtons(isLoggedIn);
    })
    .catch(error => {
      console.error('Failed to retrieve session state:', error);
    });
}

// Call getSessionState when the page loads
document.addEventListener('DOMContentLoaded', getSessionState);

// Handle logout button click
document.addEventListener('click', function(event) {
  if (event.target.id === 'logout-button' || event.target.id === 'logout-button-dropdown') {
    // Show confirmation dialog
    const confirmed = window.confirm('Are you sure you want to logout?');

    if (confirmed) {
      fetch('/logout', {
        method: 'POST'
      })
        .then(response => {
          if (response.ok) {
            return response.json();
          } else {
            throw new Error('Logout request failed.');
          }
        })
        .then(data => {
          const success = data.success;
          if (success) {
            // Redirect to home page or perform any necessary actions after logout
            window.location.href = '/';
          } else {
            // Handle logout failure
            console.error('Logout request failed:', data.message);
          }
        })
        .catch(error => {
          console.error('Logout request failed:', error);
        });
    }
  }
});


// Prompt the user to login
function promptLogin() {
  alert('Please login to proceed.'); // You can replace this with a more sophisticated login prompt
  loginContainer.classList.toggle('hidden');
  loginContainer.classList.toggle('visible');
  signupContainer.classList.add('hidden');
  signupContainer.classList.remove('visible');
}

const startButton = document.getElementById('start');
const pocContainer = document.querySelector('.poc-container');

function toggleContainer() {
  if (pocContainer.style.display === 'none') {
    pocContainer.style.display = 'block';
    document.addEventListener('click', closeContainer);
  } else {
    pocContainer.style.display = 'none';
    document.removeEventListener('click', closeContainer);
  }
}

function closeContainer(event) {
  if (!startButton.contains(event.target) && !pocContainer.contains(event.target)) {
    pocContainer.style.display = 'none';
    document.removeEventListener('click', closeContainer);
  }
}

// Check login status on "START NOW" button click
startButton.addEventListener('click', function() {
  fetch('/session_state')
    .then(response => {
      if (response.ok) {
        return response.json();
      } else {
        throw new Error('Failed to retrieve session state.');
      }
    })
    .then(data => {
      const isLoggedIn = data.isLoggedIn;
      if (!isLoggedIn) {
        promptLogin();
      } else {
        // Only toggle 'hidden' class if the user is logged in
        pocContainer.classList.toggle('hidden');
      }
    })
    .catch(error => {
      console.error('Failed to retrieve session state:', error);
    });
});


const chatbotbtn = document.getElementById('chatbot');

// Check login status on "chatbot" button click
chatbotbtn.addEventListener('click', function(event) {
  event.preventDefault(); // Prevent the default navigation behavior

  fetch('/session_state')
    .then(response => {
      if (response.ok) {
        return response.json();
      } else {
        throw new Error('Failed to retrieve session state.');
      }
    })
    .then(data => {
      const isLoggedIn = data.isLoggedIn;
      if (!isLoggedIn) {
        promptLogin();
      } else {
        // If the user is logged in, redirect to the chatbot page (replace 'chatbot.html' with the actual URL of your chatbot page)
        window.location.href = "/chatbot"; // Replace '/chatbot' with the actual chatbot page URL
      }
    })
    .catch(error => {
      console.error('Failed to retrieve session state:', error);
    });
});

document.getElementById('chapter3').addEventListener('click', function(event) {
  event.preventDefault();  // This line prevents the default action
  fetch('/get_last_unapproved')
  .then(response => response.json())
  .then(data => {
      if (data === null || !data.IMRAD) {
          window.location.href = '/genimrad';
      } else if (!data.abstract) {
          window.location.href = '/abstract';
      } else {
          window.location.href = '/fromdocx';
      }
  }).catch(error => {
      // If there's an error (like the database is empty), redirect to 'fromdocx'
      window.location.href = '/fromdocx';
  });
});



