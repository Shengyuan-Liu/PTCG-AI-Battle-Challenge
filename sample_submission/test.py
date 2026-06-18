from kaggle_environments import make
from main import agent

with open("deck.csv") as f:
    deck = [int(line) for line in f.readlines() if line.strip()]

env = make("cabt", configuration={"decks": [deck, deck]})
env.run([agent, agent])

with open("result.html", "w") as f:
    f.write(env.render(mode="html"))

print("Simulation finished.")