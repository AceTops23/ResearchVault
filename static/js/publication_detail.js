function togglePdfViewer() {
    var pdfViewer = document.getElementById('pdfViewer');
    pdfViewer.style.display = (pdfViewer.style.display === 'none') ? 'block' : 'none';
}

function convertToImrad(item_id) {
    fetch(`/convert_to_imrad/${item_id}`)
        .then(response => response.blob()) // Convert response to a Blob
        .then(blob => {
            // Create a new Blob URL for the converted PDF
            const pdfUrl = URL.createObjectURL(blob);

            // Update the embedded PDF viewer to show the converted PDF
            const pdfViewer = document.getElementById('pdfViewer');
            pdfViewer.setAttribute('data', pdfUrl);
            pdfViewer.style.display = 'block';
        })
        .catch(error => {
            console.error('Error:', error);
        });
}


function generateAPACitation() {
    var item_id = {{ item.id }};
    console.log("Item ID:", item_id); // Debug message

    fetch(`/generate_apa_citation/${item_id}`)
        .then(response => response.json())
        .then(data => {
            console.log("Response:", data); // Debug message

            if (data.apa_citation) {
                showAPACitation(data.apa_citation);
            } else {
                alert("Publication not found.");
            }
        })
        .catch(error => {
            console.error('Error fetching APA citation:', error);
            alert("Error fetching APA citation.");
        });
}


function showAPACitation(citation) {
    var container = document.getElementById('apa-citation-container');
    container.innerText = citation;
    container.style.display = 'block';
}


function closeAPACitation() {
    var citationContainer = document.getElementById("apa-citation-container");
    citationContainer.innerHTML = "";
    citationContainer.classList.add("hidden");
}

document.addEventListener("click", function (event) {
    if (!event.target.closest("#apa-citation-container") && !event.target.matches("#apa-button")) {
        closeAPACitation();
    }
});

function pdfViewer(file) {
    const reader = new FileReader();
    const documentPreview = document.getElementById("pdfViewer");
    documentPreview.innerHTML = ""; // Clear previous content

    reader.onprogress = function (event) {
        if (event.lengthComputable) {
            const percentLoaded = (event.loaded / event.total) * 100;
            updateLoadingPercentage(percentLoaded);
        }
    };    
}