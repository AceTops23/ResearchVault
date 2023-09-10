
var quill = new Quill('#editor', {
    modules: {
      toolbar: '#toolbar', 
    },
    theme: 'snow'
  });
  
 const comment = document.getElementById('comment');
 const commentsection = document.getElementById('comment-section');
 const commentexit = document.getElementById('comment-exit');

 comment.addEventListener('click', function() {
    if (commentsection.style.display === 'none') {
        commentsection.style.display = 'block'; 
    } else {
        commentsection.style.display = 'none'; 
    }
  });

commentexit.addEventListener('click', function(){
    commentsection.style.display = 'none';
})

function addComment() {
    const commentInput = document.querySelector('#commentInput').value;
    const selectedText = getSelectedText();

    if (!selectedText) {
        alert("Please select a word or phrase to comment on.");
        return;
    }

    // Create a comment element with both the selected text and the comment
    const commentElement = document.createElement('div');
    commentElement.innerHTML = `<strong>Comment on "${selectedText}":</strong> ${commentInput}`;

    const range = window.getSelection().getRangeAt(0);
    console.log('Range:', range);
    const span = document.createElement('span');
    span.className = 'highlighted';
    range.surroundContents(span);

    // Append the comment element to the comments section
    document.querySelector('#comments').appendChild(commentElement);

    // Clear the comment input field
    document.querySelector('#commentInput').value = '';
}

function getSelectedText() {
    const selection = window.getSelection();
    if (selection.rangeCount > 0) {
        return selection.toString();
    }
    return null;
}