import requests
from update_density import update_density, level

# Updating user submission to set density level
update = max(update_density(), level)
density_level =  update if update else 35
if density_level < 10 or density_level > 100:
    density_level = 35
    print("Please choose a density_level between 10 and 100 to optimise processing power for this demo version." )
    print("Layers density initialised at 35 by default.")
else: print("layers density succesfully set at{number}")

"""""
# This function is to be implemented on the frontend user interface
def get_suggestion(number):
    url = "local_endpoint/{}".format(number)
        
    r = requests.get(url)
    if r.status_code == 200:
        print("layers density succesfully set at{number}")
    else:
        print("An error occurred, code={}".format(r.status_code))
        print("Hint: For a good startpoint try again with a positive integer between 10 and 100 (Ex: 35 by default)")
    density_level = number 
"""""
