<?php
// Using a JavaScript if-statement to trigger the action
echo "Submitting Layers Density value for backend update.";

?>

<script type="text/javascript">
  if ($_SERVER['REQUEST_METHOD'] === 'POST') {
      // logic of the action after the setting has been submitted by the user     
      document.getElementById("update").addEventListener("submit", () => {
        fetch("/src/api.py", { method: "POST" }) 
        .then(response => response.text()) 
        .then(data => alert(data)) 
        .catch(error => console.error("Error:", error));
        alert("Density level submitted successfully!");
        alert("JavaScript executed via PHP script!");
        alert("Get set and ready for the next phase of this presention!");    
      });
    
  } 
</script>

