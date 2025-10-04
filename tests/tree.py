import loader
from core.datatypes.sequence import Sequence

import services.inspector as inspector
import services.agg as agg
import services.client as client

#item = inspector.Inspector.get_item(45467717)

#top = agg.PAggregator.get_jobs()

top = client.HNClient.best()

print(f"# Kids: {top.kids.length()}")

curr = top
descendants = Sequence([curr.kids])

PASSES = 3

for i in range(PASSES):
    curr = client.HNClient.expand(curr.kids)
    descendants.append(
        curr.kids
    )

print(f"{PASSES} Passes: {descendants.flatten().length()}")
