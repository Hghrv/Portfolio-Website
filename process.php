<?php
// Using a JavaScript if-statement to trigger the action
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    // logic of the action after the setting has been submitted by the user
    echo "Submitting Layers Density value for backend update.')";  
    document.getElementById("update").addEventListener("click", () => {
        fetch("/src/api.py", { method: "POST" })
          .then(response => response.text())
          .then(data => alert(data))
          .catch(error => console.error("Error:", error));
      });
    echo "Density level submitted successfully!";
    echo "PHP script executed via JavaScript!";
    echo "Get set and ready for the next phase of this presention!";    
}

// Semingly best common practise to not include a php closing tag anymore
// end of script