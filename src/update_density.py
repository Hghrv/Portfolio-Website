from flask import Flask, request

app = Flask(__name__)

# Decorating a submit_density() function with the form's action and method
@app.route('/submit', methods=['POST'])
def update_density():
    density = request.form['density']  # Accessing form density from user input
    return density

#  Enabling detailed error tracebacks during development on main module.

if __name__ == '__main__':
    app.run(debug=True)