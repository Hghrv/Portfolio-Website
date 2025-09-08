<?php
// Using a JavaScript if-statement to trigger the action
if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['myButton'])) {
    // logic of the action after the setting has been submitted by the user
    echo "Density level submitted successfully!";
    echo "PHP script executed via JavaScript!";
    echo "Get set and ready for the next phase of this presention!";
}
?>