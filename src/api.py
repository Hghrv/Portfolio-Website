import requests

def get_suggestion(number):
    url = "local_endpoint/{}".format(number)
        
    r = requests.get(url)
    if r.status_code == 200:
        print("layers density succesfully set at{number}")
    else:
        print("An error occurred, code={}".format(r.status_code))
        print("Hint: For a good startpoint try again with a positive integer between 10 and 100 (Ex: 35)")
