
var quill = new Quill('#editor', {
    modules: {
      toolbar: '#toolbar', // Pass the toolbar container selector
    },
    theme: 'snow'
  });
  
 const comment = document.getElementById('comment');
 const commentsection = document.getElementById('comment-section');
 const commentexit = document.getElementById('comment-exit');

 comment.addEventListener('click', function() {
    if (commentsection.style.display === 'none') {
        commentsection.style.display = 'block'; // Show the div
    } else {
        commentsection.style.display = 'none'; // Hide the div
    }
  });

commentexit.addEventListener('click', function(){
    commentsection.style.display = 'none';
})