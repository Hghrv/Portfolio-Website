import requests

def get_suggestion(number):
    url = "local_endpoint/{}".format(number)
        
    r = requests.get(url)
    if r.status_code == 200:
        print(r.text)
    else:
        print("An error occurred, code={}".format(r.status_code))
