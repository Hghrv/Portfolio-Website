from flask import Flask, request, render_template, jsonify

# Default density
level = 35

# initialising the app
app = Flask(__name__)

# Targetting the webpage
@app.route('/')
def index():
    return render_template('projects.html')

# Decorating a submit_density() function with the form's action and method
@app.route('/submit', methods=['POST'])
def update_density():
    density = request.form['density']  # Accessing form density from user input
    level = density
    return density

#  Enabling detailed error tracebacks during development on main module.
if __name__ == '__main__':
    app.run(debug=True)