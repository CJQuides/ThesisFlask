{% extends "base.html" %} {% block title %}Predict Me{% endblock %} {% block
content %}

    <form name="form" action="/form" method="post">
        
        <label for="quantity">Quantity (between 1 and 5):</label>
        <input type="number" id="quantity" name="rateA" min="1" max="5">
        
        <label for="quantity">Quantity (between 1 and 5):</label>
        <input type="number" id="quantity" name="rateB" min="1" max="5">

        <br><br><br>
        <div style="" id="formButtons">
            <input type="submit" id="submitButton" class="btn btn-success" value="Submit">
            <button type="button" onclick="closeBtn()" style="border-radius: 25px; padding: 0 25px;">Click Me</button>
            <!-- <a href="index.php" id="cancelButton" class="btn btn-danger ml-2">Cancel</a> -->
        </div>
        
    </form>
    {{ pred }} {{userName}}

    <!-- <button id="subBtn" onclick="checker()" class="float-end">Click Me</button> -->
    <!-- <button type="button" id="subBtn" onclick="checker()" class="float-end">Click Me</button> -->
    
{% endblock %}