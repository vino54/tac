import resquests
import json 
def fetch_football_legends(api_key):
    endpoint = "https://www.thesportsdb.com/api/v1/json/1/searchplayers.php"
    
    # Vous pouvez spécifier le nom de la légende que vous recherchez dans les paramètres.
    params = {
        "p": "Pelé"  # Remplacez "Pelé" par le nom du joueur que vous recherchez.
    }

    response = requests.get(endpoint, params=params)

    if response.status_code == 200:
        data = response.json()
        players = data.get("player")
        if players:
            for player in players:
                name = player.get("strPlayer")
                print(f"Nom du joueur : {name}")
        else:
            print("Aucun joueur trouvé.")
    else:
        print(f"Une erreur s'est produite : {response.status_code}")

api_key = "votre_clé_API"
fetch_football_legends(api_key)
